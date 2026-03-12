# ☕ Coffee Bean Quality Grading using Machine Learning

## 📌 Project Overview

This project implements an AI-based coffee bean quality grading system using texture feature extraction and machine learning models.

The system classifies coffee beans into four quality grades:

* **CGA** → Grade A (Premium)
* **CGB** → Grade B (Good)
* **CGC** → Grade C (Standard)
* **CGD** → Grade D (Defective)

The final model will be integrated into a full-stack web application for real-time prediction.

---

## 🎯 Objectives

* Automate coffee bean quality grading
* Extract texture features using **GLCM and LBP**
* Compare multiple machine learning models
* Deploy the best-performing model in a full-stack system

---

## 🧠 Machine Learning Pipeline

Dataset
→ Preprocessing (Resizing, Grayscale Conversion, Normalization)
→ Texture Feature Extraction (GLCM + LBP)
→ Train-Test Split
→ Model Training (Random Forest, SVM, KNN)
→ Performance Evaluation
→ Best Model Selection
→ Integration with Flask Web Application

---

## 🛠️ Tech Stack

### 🔹 Machine Learning

* Python
* NumPy
* OpenCV
* Scikit-image
* Scikit-learn

### 🔹 Backend

* Flask

### 🔹 Database

* SQLite

### 🔹 Frontend

* HTML
* CSS
* JavaScript

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📂 Project Structure

```
coffee-quality-grading/
│
├── dataset/
├── ml/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── train_models.py
│
├── backend/
├── frontend/
├── database/
├── requirements.txt
└── README.md
```

---

## 👥 Team Collaboration

This project uses Git and GitHub for version control and collaborative development.

---

## 🚀 Future Work

* Hyperparameter tuning
* Model optimization
* Deployment on cloud
* Enhanced UI for grading visualization

---

## 📌 Status

🔄 Currently in development (ML pipeline implementation phase)
