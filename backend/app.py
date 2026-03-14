from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import uuid

# =========================
# Absolute Paths (Professional Setup)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "best_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "..", "database", "coffee.db")

print("Working Directory:", os.getcwd())
print("Database Path:", DB_PATH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# =========================
# Load ML Model
# =========================
saved = joblib.load(MODEL_PATH)
model = saved["model"]
label_encoder = saved["label_encoder"]
scaler = saved.get("scaler", None)

# =========================
# Flask App Init
# =========================
app = Flask(
    __name__,
    static_folder="../frontend",
    template_folder="../frontend"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# =========================
# Initialize Database
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_grade TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# Recommendation System
# =========================
def get_recommendation(grade):
    if grade == "CGA":
        return "Premium beans suitable for export."
    elif grade == "CGB":
        return "Good quality, recommended for commercial roasting."
    elif grade == "CGC":
        return "Standard quality, suitable for blended coffee."
    else:
        return "Defective beans detected. Recommended for removal."

# =========================
# Image Preprocessing
# =========================
def preprocess_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image.")
    img = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

# =========================
# Feature Extraction
# =========================
def extract_features(img):
    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    lbp = local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)

    features = np.array([
        contrast,
        energy,
        homogeneity,
        correlation,
        lbp_mean,
        lbp_std
    ])

    return features.reshape(1, -1), {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation,
        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std
    }

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        features_array, feature_values = extract_features(img)

        if scaler is not None:
            features_array = scaler.transform(features_array)

        pred = model.predict(features_array)
        proba = model.predict_proba(features_array)

        grade = label_encoder.inverse_transform(pred)[0]
        confidence = round(np.max(proba) * 100, 2)
        recommendation = get_recommendation(grade)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO grades (filename, predicted_grade, confidence, timestamp) VALUES (?, ?, ?, ?)",
            (
                unique_name,
                grade,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        conn.commit()
        conn.close()

        # Save features for PDF
        np.save(os.path.join(UPLOAD_FOLDER, f"{unique_name}_features.npy"), feature_values)

        print("Inserted:", grade, confidence)

        return jsonify({
            "grade": grade,
            "confidence": confidence,
            "recommendation": recommendation,
            "features": feature_values,
            "filename": unique_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, predicted_grade, confidence, timestamp FROM grades ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    return render_template("history.html", records=rows)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM grades")
    total = c.fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM grades WHERE timestamp LIKE ?", (f"{today}%",))
    today_count = c.fetchone()[0]

    c.execute("SELECT predicted_grade, COUNT(*) FROM grades GROUP BY predicted_grade")
    rows = c.fetchall()
    conn.close()

    grade_counts = {grade: count for grade, count in rows}

    return render_template(
        "dashboard.html",
        total=total,
        today_count=today_count,
        grade_counts=grade_counts
    )

@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html")

    files = request.files.getlist("files")
    results = []
    summary = {}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for file in files:
        if file.filename == "":
            continue

        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        try:
            img = preprocess_image(filepath)
            features_array, feature_values = extract_features(img)

            if scaler is not None:
                features_array = scaler.transform(features_array)

            pred = model.predict(features_array)
            proba = model.predict_proba(features_array)

            grade = label_encoder.inverse_transform(pred)[0]
            confidence = round(np.max(proba) * 100, 2)

            # Save to DB
            c.execute(
                "INSERT INTO grades (filename, predicted_grade, confidence, timestamp) VALUES (?, ?, ?, ?)",
                (
                    unique_name,
                    grade,
                    confidence,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

            # Save features for PDF
            np.save(os.path.join(UPLOAD_FOLDER, f"{unique_name}_features.npy"), feature_values)

            results.append({
                "filename": file.filename,
                "grade": grade,
                "confidence": confidence
            })

            summary[grade] = summary.get(grade, 0) + 1
        except:
            continue

    conn.commit()
    conn.close()
    return render_template("batch.html", results=results, summary=summary)

# =========================
# PDF Export (Single Button)
# =========================
@app.route("/export/pdf/<filename>")
def export_pdf(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT predicted_grade, confidence, timestamp FROM grades WHERE filename=?",
        (filename,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return "Record not found"

    grade, confidence, timestamp = row
    recommendation = get_recommendation(grade)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}.pdf")
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Coffee Quality Inspection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Inspection ID: {filename}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {timestamp}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    if os.path.exists(file_path):
        img = Image(file_path, width=3*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3 * inch))

    # Table: Prediction + Recommendation + Texture Features
    data = [
        ["Parameter", "Value"],
        ["Predicted Grade", grade],
        ["Confidence", f"{confidence}%"],
        ["Recommendation", recommendation]
    ]

    # Include saved features
    feature_file = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}_features.npy")
    if os.path.exists(feature_file):
        features = np.load(feature_file, allow_pickle=True).item()
        for key, value in features.items():
            data.append([key.capitalize(), f"{value:.4f}"])

    table = Table(data, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Inspector Signature: ____________________", styles["Normal"]))

    doc.build(elements)
    return send_from_directory(app.config["UPLOAD_FOLDER"], f"{filename}.pdf", as_attachment=True)

@app.route("/knowledge")
def knowledge():
    return render_template("knowledge.html")

# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)