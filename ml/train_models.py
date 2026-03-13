import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import time

# Paths
FEATURES_CSV = "ml/features.csv"
BEST_MODEL_PATH = "ml/best_model.pkl"

# 1️⃣ Load features
df = pd.read_csv(FEATURES_CSV)

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels (CGA, CGB, CGC, CGD → 0,1,2,3)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2️⃣ Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3️⃣ Feature scaling (for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

# 5️⃣ Train and evaluate models
for name, model in models.items():
    print(f"\n⏳ Training {name}...")
    start_time = time.time()
    
    # Random Forest uses unscaled features
    if name == "RandomForest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    elapsed = time.time() - start_time
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    
    print(f"✅ {name} trained in {elapsed:.2f} seconds")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# 6️⃣ Select best model by accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]

# Save best model
if best_model_name in ["SVM", "KNN"]:
    joblib.dump({"model": best_model, "scaler": scaler, "label_encoder": le}, BEST_MODEL_PATH)
else:
    joblib.dump({"model": best_model, "label_encoder": le}, BEST_MODEL_PATH)

print(f"\n🏆 Best Model: {best_model_name} saved to {BEST_MODEL_PATH}")