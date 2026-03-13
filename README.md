
# Heart Disease Detection using Machine Learning

<div align="center">
  
  ![Heart Disease](https://img.shields.io/badge/Healthcare-ML-red)
  ![Python](https://img.shields.io/badge/Python-3.10-blue)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange)
  ![Docker](https://img.shields.io/badge/Docker-Ready-blue)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  
  ### ❤️ Predicting Heart Disease with 97.7% Recall using Machine Learning
  
  [Overview](#overview) • 
  [Features](#-key-features) • 
  [Tech Stack](#-tech-stack) • 
  [Installation](#-installation-guide) • 
  [Usage](#-how-to-use) • 
  [API](#-api-endpoints) • 
  [Results](#-model-performance) • 
  [Screenshots](#-screenshots) • 
  [Contributing](#-contributing)
  
</div>

---

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [✨ Key Features](#-key-features)
- [📊 Dataset](#-dataset)
- [🔬 Features](#-features)
- [💻 Tech Stack](#-tech-stack)
- [📥 Installation Guide](#-installation-guide)
- [🚀 How to Run](#-how-to-run)
- [🎯 How to Use](#-how-to-use)
- [🌐 API Endpoints](#-api-endpoints)
- [📈 Model Performance](#-model-performance)
- [🖼️ Screenshots](#%EF%B8%8F-screenshots)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🧪 Testing](#-testing)
- [🐳 Docker Deployment](#-docker-deployment)
- [📊 Results](#-results)
- [🔮 Future Scope](#-future-scope)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📧 Contact](#-contact)
- [🙏 Acknowledgments](#-acknowledgments)

---

## Overview

The **Heart Disease Detection System** is an end-to-end machine learning web application that predicts the likelihood of heart disease in patients based on their clinical parameters. Built with a **Random Forest classifier** achieving **97.7% recall**, this system provides healthcare professionals with a reliable screening tool for early detection of cardiovascular diseases.

The project demonstrates a complete ML pipeline from data preprocessing to model deployment, featuring:
- Multiple classification algorithms (Random Forest, Logistic Regression, SVM, Decision Tree)
- Optimized threshold selection for maximum recall
- Interactive web interface built with Streamlit
- RESTful API using FastAPI
- Real-time predictions with explainability features

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide, accounting for approximately 17.9 million deaths annually according to WHO. Early and accurate detection through diagnostic screening can significantly improve patient survival rates and reduce healthcare costs.

This project addresses the critical need for accessible, accurate, and interpretable heart disease prediction by building machine learning models that analyze patients' demographic, clinical, and diagnostic test data, enabling healthcare professionals to make timely and informed decisions.

---

## Objectives

- ✅ Develop and compare multiple classification algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest ✓ (Best Performer)
  - Support Vector Machine (SVM)
- ✅ Achieve high **Recall (Sensitivity)** to minimize false negatives
- ✅ Optimize decision threshold for balanced performance
- ✅ Deploy the best model as a REST API
- ✅ Create an interactive user interface for easy use
- ✅ Provide model interpretability through feature importance
- ✅ Deliver actionable insights with risk factor analysis

---

## ✨ Key Features

### 1. **Multiple ML Models**
- Implementation of 4 classification algorithms
- Comprehensive model comparison with metrics
- Automatic selection of best performing model

### 2. **Threshold Optimization**
- Dynamic threshold testing (0.20 to 0.65)
- Optimal threshold: **0.37**
- Balanced recall (90.9%) and precision (62.5%)

### 3. **Feature Importance Visualization**
- Interactive horizontal bar chart
- Shows top influencing factors:
  - Max Heart Rate (17.7%)
  - Age (14.9%)
  - Cholesterol (13.9%)
  - Resting BP (13.0%)
  - ST Depression (10.8%)

### 4. **Risk Factors Analysis**
- Real-time identification of patient risk factors
- Color-coded severity levels (🔴 High, 🟡 Moderate, 🟢 Normal)
- Personalized health insights

### 5. **Professional UI/UX**
- Medical-grade gradient design
- Responsive layout for all devices
- Animated results with fade-in effects
- Color-coded risk boxes (🔴 High, 🟡 Moderate, 🟠 Borderline, 🟢 Low)

### 6. **Real-time Predictions**
- Instant results with probability scores
- Confidence level indicators
- Actionable health recommendations

### 7. **API Integration**
- FastAPI backend with comprehensive documentation
- Automatic Swagger UI at `/docs`
- Health check endpoint

### 8. **Model Interpretability**
- Feature importance analysis
- SHAP-ready architecture
- Transparent decision making

### 9. **Dataset Statistics**
- Training data transparency
- Class distribution visualization
- Source attribution

### 10. **Comprehensive Error Handling**
- API connection status indicator
- Graceful failure messages
- User-friendly error displays

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | `heart_disease_dataset.csv` (UCI Repository) |
| **Records** | 400 patients |
| **Features** | 13 clinical features |
| **Target** | `heart_disease` (0 = No Disease, 1 = Disease) |
| **Missing Values** | None (Clean dataset) |
| **Data Type** | All numerical (no encoding needed) |
| **Class Distribution** | Class 0: 235 (58.75%), Class 1: 165 (41.25%) |

---

## 🔬 Features

| # | Feature | Description | Range | Clinical Significance |
|---|---------|-------------|-------|----------------------|
| 1 | `age` | Age in years | 29-77 | Risk increases with age |
| 2 | `sex` | Gender (0=Female, 1=Male) | 0,1 | Males at higher risk |
| 3 | `chest_pain_type` | Type of chest pain | 0-3 | Indicator of angina |
| 4 | `resting_blood_pressure` | Resting BP (mm Hg) | 94-200 | Hypertension indicator |
| 5 | `cholesterol` | Serum cholesterol (mg/dl) | 126-564 | Key risk factor |
| 6 | `fasting_blood_sugar` | FBS >120 mg/dl (0=No, 1=Yes) | 0,1 | Diabetes indicator |
| 7 | `resting_ecg` | Resting ECG results | 0-2 | Heart electrical activity |
| 8 | `max_heart_rate` | Maximum heart rate achieved | 71-202 | Exercise capacity |
| 9 | `exercise_induced_angina` | Exercise angina (0=No, 1=Yes) | 0,1 | Stress test indicator |
| 10 | `st_depression` | ST depression induced by exercise | 0-6.2 | Ischemia indicator |
| 11 | `st_slope` | Slope of ST segment | 0-2 | Heart stress response |
| 12 | `num_major_vessels` | Number of major vessels | 0-3 | Blockage indicator |
| 13 | `thalassemia` | Thalassemia type | 0-3 | Blood disorder |

---

## 💻 Tech Stack

### **Core Libraries**
```txt
pandas==2.0.3              # Data manipulation
numpy==1.24.3              # Numerical computations
matplotlib==3.7.1          # Data visualization
seaborn==0.12.2            # Statistical plotting
scikit-learn==1.3.0        # Machine learning algorithms
scipy==1.10.1              # Scientific computing
```

### **Backend (API)**
```txt
fastapi==0.104.1           # REST API framework
uvicorn==0.24.0            # ASGI server
pydantic==2.5.0            # Data validation
joblib==1.3.2              # Model serialization
python-multipart==0.0.6    # Form data handling
```

### **Frontend**
```txt
streamlit==1.28.1          # Web application framework
plotly==5.17.0             # Interactive visualizations
plotly-express==0.4.1      # Simplified plotting
requests==2.31.0           # HTTP requests
```

### **Development & Deployment**
```txt
docker==24.0.6             # Containerization
docker-compose==2.21.0     # Multi-container orchestration
pytest==7.4.0              # Testing framework
black==23.9.1              # Code formatting
```

---

## 📥 Installation Guide

### **Prerequisites**
- Python 3.10 or higher
- pip package manager
- Git (optional)
- 4GB RAM minimum
- 2GB free disk space

### **Step-by-Step Installation**

#### **Step 1: Clone or Extract Project**
```bash
# If using git:
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection

# If using zip file:
# Extract the zip file and navigate to the folder
cd path/to/extracted/Heart-Disease-Detection
```

#### **Step 2: Create Virtual Environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

*You should see `(venv)` in your terminal prompt*

#### **Step 3: Upgrade pip**
```bash
python -m pip install --upgrade pip
```

#### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

*This will install all required packages (takes 2-3 minutes)*

#### **Step 5: Verify Installation**
```bash
python --version
pip list
```

#### **Step 6: Train the Model (Optional - models are pre-trained)**
```bash
python src/model_training.py
python src/optimize_threshold.py
```

---

## 🚀 How to Run

### **Method 1: Local Development (Two Terminals)**

#### **Terminal 1 - Start Backend API:**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Start FastAPI server
python src/api.py
```
**Expected Output:**
```
============================================================
🚀 HEART DISEASE DETECTION API v2.0
============================================================
✅ Loaded ORIGINAL model
📊 Model Configuration:
   • Model Type: RandomForestClassifier
   • Optimized Threshold: 0.37
   • Expected Recall: 90.9%
   • Expected Precision: 62.5%
============================================================
INFO: Uvicorn running on http://0.0.0.0:8000
```
⚠️ **Keep this terminal open**

#### **Terminal 2 - Start Frontend (New Terminal):**
```bash
# Navigate to project folder
cd path/to/Heart-Disease-Detection

# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Start Streamlit app
streamlit run app.py
```

#### **Access the Application:**
- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### **Method 2: Docker Deployment (One Command)**
```bash
docker-compose up --build
```

---

## 🎯 How to Use

### **Step-by-Step User Guide**

#### **1. Enter Patient Information**
Fill in all 13 clinical parameters:
- **Basic Information** (Left column):
  - Age (years)
  - Sex (Female/Male)
  - Chest Pain Type (0-3)
  - Resting Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  
- **Clinical Measurements** (Right column):
  - Resting ECG
  - Max Heart Rate
  - Exercise Induced Angina
  - ST Depression
  - ST Slope
  - Number of Major Vessels
  - Thalassemia

#### **2. Get Prediction**
- Click **"🔍 Predict Heart Disease Risk"** button
- Wait for the spinner animation

#### **3. Interpret Results**

**Risk Level Colors:**
- 🔴 **HIGH RISK** (>70%) - Immediate action required
- 🟡 **MODERATE RISK** (50-70%) - Schedule appointment
- 🟠 **BORDERLINE** (30-50%) - Monitor closely
- 🟢 **LOW RISK** (<30%) - Routine check-ups

**Result Components:**
- Large probability percentage
- Progress bar visualization
- Confidence level badge
- Model threshold information
- Personalized message
- Recommended action

#### **4. Explore Additional Insights**

**Feature Importance Chart:**
- Shows which factors most influenced the prediction
- Horizontal bar chart with percentages
- Color-coded by importance

**Risk Factors Analysis:**
- Highlights specific risk factors
- Color-coded severity levels
- Detailed explanations

**Key Health Metrics:**
- Age
- Cholesterol level with status
- Max Heart Rate with status

#### **5. Review Sidebar Information**

- **Model Performance Metrics**
- **Model Comparison Dashboard**
- **Risk Level Guide**
- **Dataset Statistics**
- **Medical Disclaimer**

---

## 🌐 API Endpoints

### **Base URL**
```
http://localhost:8000
```

### **1. Root Endpoint**
```http
GET /
```
**Response:**
```json
{
  "message": "Heart Disease Detection API",
  "status": "active",
  "threshold": 0.37
}
```

### **2. Health Check**
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.37
}
```

### **3. Model Information**
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "threshold": 0.37,
  "features": ["age", "sex", "chest_pain_type", ...],
  "feature_importance": {...}
}
```

### **4. Prediction Endpoint**
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 58,
  "sex": 1,
  "chest_pain_type": 1,
  "resting_blood_pressure": 134,
  "cholesterol": 246,
  "fasting_blood_sugar": 0,
  "resting_ecg": 0,
  "max_heart_rate": 155,
  "exercise_induced_angina": 0,
  "st_depression": 0.4,
  "st_slope": 1,
  "num_major_vessels": 1,
  "thalassemia": 2
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "High Risk",
  "message": "Patient shows signs of heart disease (Risk: High Risk)",
  "threshold": 0.37
}
```

### **5. API Documentation**
```http
GET /docs
```
Opens interactive Swagger UI for API testing.

---

## 📈 Model Performance

### **Default Threshold (0.5) Comparison**

| Model | Recall | Precision | F1-Score | Accuracy | ROC-AUC |
|-------|--------|-----------|----------|----------|---------|
| **Random Forest** | **97.7%** | 56.6% | 0.717 | 67.5% | 0.7617 |
| Logistic Regression | 90.9% | 62.5% | 0.740 | 67.5% | 0.7342 |
| SVM | 88.6% | 62.9% | 0.735 | 65.0% | 0.7355 |
| Decision Tree | 75.0% | 60.0% | 0.667 | 57.5% | 0.5631 |

### **Optimized Threshold (0.37)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | **90.9%** | Catches 40/44 disease cases |
| **Precision** | 62.5% | Acceptable false positives |
| **F1-Score** | 0.741 | Balanced performance |
| **False Negatives** | 4 | Missed patients (vs 11 at default) |
| **False Positives** | 24 | False alarms (vs 33 at 0.30) |

### **Confusion Matrix (at 0.37 threshold)**
```
                Actual
              No    Yes
Predicted No   21     4
Predicted Yes  24    40
```

### **Feature Importance**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Max Heart Rate | 17.7% |
| 2 | Age | 14.9% |
| 3 | Cholesterol | 13.9% |
| 4 | Resting Blood Pressure | 13.0% |
| 5 | ST Depression | 10.8% |
| 6 | ST Slope | 6.9% |
| 7 | Thalassemia | 6.2% |
| 8 | Chest Pain Type | 5.4% |
| 9 | Major Vessels | 4.3% |
| 10 | Resting ECG | 3.0% |
| 11 | Sex | 2.2% |
| 12 | Exercise Angina | 1.6% |
| 13 | Fasting Blood Sugar | 0.9% |

---

## 🖼️ Screenshots

### **Main Dashboard**
```
┌─────────────────────────────────────────────────────────────┐
│  ❤️ Heart Disease Detection System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │ Patient Information  │    │   Prediction Results │      │
│  │                      │    │   ┌───────────────┐  │      │
│  │ Age: 58             │    │   │ 🔴 HIGH RISK   │  │      │
│  │ Sex: Male           │    │   │    85.3%       │  │      │
│  │ Chest Pain: 1       │    │   └───────────────┘  │      │
│  │ BP: 134             │    │                      │      │
│  │ Cholesterol: 246    │    │   • Feature Importance │      │
│  │ ...                 │    │   • Risk Factors      │      │
│  └──────────────────────┘    └──────────────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Sidebar: Model Performance • Risk Guide • Dataset   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### **Feature Importance Chart**
```
🔍 Feature Importance (What influences prediction most?)

Max Heart Rate     ■■■■■■■■■■■■■■■■■ 17.7%
Age                ■■■■■■■■■■■■■■■ 14.9%
Cholesterol        ■■■■■■■■■■■■■ 13.9%
Resting BP         ■■■■■■■■■■■■ 13.0%
ST Depression      ■■■■■■■■■■ 10.8%
...                ...
```

### **Risk Factors Analysis**
```
⚠️ Your Risk Factors Analysis

🔴 Age > 55         Your age (58) increases cardiovascular risk
🟡 Elevated BP      134 mm Hg
🟢 Normal Cholesterol 180 mg/dl
```

---

## 📁 Project Structure

```
HEART-DISEASE-DETECTION/
│
├── 📂 data/                          # Dataset storage
│   ├── heart_disease_dataset.csv      # Original data (400 records)
│   └── 📂 processed/                  # Preprocessed data
│       ├── X_train.csv                # Scaled training features
│       ├── X_test.csv                 # Scaled test features
│       ├── y_train.csv                # Training labels
│       └── y_test.csv                 # Test labels
│
├── 📂 models/                         # Saved models & artifacts
│   ├── best_model.pkl                 # Random Forest model (production)
│   ├── scaler.pkl                      # StandardScaler for preprocessing
│   ├── threshold.pkl                    # Optimal threshold (0.37)
│   ├── model_comparison.csv              # All model metrics
│   ├── confusion_matrix.png               # Visual confusion matrix
│   ├── roc_curve.png                       # ROC curve plot
│   ├── feature_importance.png               # Feature importance chart
│   └── threshold_analysis.png                # Threshold vs metrics plot
│
├── 📂 src/                            # Source code
│   ├── api.py                          # FastAPI backend (PREDICTION LOGIC)
│   ├── model_training.py                # Training script (MODEL BUILDING)
│   ├── evaluation.py                     # Model evaluation & metrics
│   ├── optimize_threshold.py              # Find best threshold
│   ├── test_api.py                        # API testing
│   └── utils.py                           # Helper functions
│
├── app.py                               # Streamlit frontend (MAIN UI)
├── EDA.ipynb                            # Exploratory Data Analysis
├── requirements.txt                     # Dependencies list
├── Dockerfile                           # Docker configuration
├── docker-compose.yml                    # Multi-container setup
├── .dockerignore                         # Files to exclude from Docker
├── test_model.py                         # Quick model test
├── README.md                             # Project documentation
└── .gitignore                            # Git ignore file
```

---

## ⚙️ Configuration

### **Environment Variables**
Create a `.env` file (optional):
```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/best_model.pkl
SCALER_PATH=models/scaler.pkl
THRESHOLD_PATH=models/threshold.pkl
```

### **Model Configuration**
Edit `src/api.py` to modify:
```python
# Threshold adjustment
THRESHOLD = 0.37  # Change this value

# API host/port
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port if needed
```

---

## 🧪 Testing

### **Run API Tests**
```bash
python src/test_api.py
```

### **Test Model Loading**
```bash
python test_model.py
```

### **Manual API Test with Curl**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"chest_pain_type":1,"resting_blood_pressure":134,"cholesterol":246,"fasting_blood_sugar":0,"resting_ecg":0,"max_heart_rate":155,"exercise_induced_angina":0,"st_depression":0.4,"st_slope":1,"num_major_vessels":1,"thalassemia":2}'
```

### **Test All Test Cases**
```bash
python -c "
import requests
import json

test_cases = [
    {'name': 'Low Risk', 'data': {'age':32,'sex':0,'chest_pain_type':0,'resting_blood_pressure':110,'cholesterol':160,'fasting_blood_sugar':0,'resting_ecg':0,'max_heart_rate':175,'exercise_induced_angina':0,'st_depression':0.1,'st_slope':1,'num_major_vessels':0,'thalassemia':0}},
    {'name': 'Moderate Risk', 'data': {'age':55,'sex':1,'chest_pain_type':2,'resting_blood_pressure':140,'cholesterol':240,'fasting_blood_sugar':1,'resting_ecg':1,'max_heart_rate':135,'exercise_induced_angina':1,'st_depression':1.5,'st_slope':2,'num_major_vessels':1,'thalassemia':2}},
    {'name': 'High Risk', 'data': {'age':72,'sex':1,'chest_pain_type':3,'resting_blood_pressure':170,'cholesterol':320,'fasting_blood_sugar':1,'resting_ecg':2,'max_heart_rate':100,'exercise_induced_angina':1,'st_depression':3.5,'st_slope':2,'num_major_vessels':3,'thalassemia':3}}
]

for test in test_cases:
    response = requests.post('http://localhost:8000/predict', json=test['data'])
    result = response.json()
    print(f"{test['name']}: {result['probability']*100:.1f}% - {result['risk_level']}")
"
```

---

## 🐳 Docker Deployment

### **Prerequisites**
- Docker Desktop installed
- Docker Compose installed

### **Dockerfile**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["sh", "-c", "python src/api.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
```

### **docker-compose.yml**
```yaml
version: '3.8'

services:
  heart-disease-app:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
```

### **Deploy Commands**
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f
```

### **Access Deployed App**
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📊 Results

### **Key Achievements**

| Achievement | Result |
|-------------|--------|
| **Highest Recall** | 97.7% (Random Forest) |
| **Optimized Recall** | 90.9% (Balanced) |
| **False Positives Reduced** | 33 → 24 (27% improvement) |
| **Precision Improvement** | 56.6% → 62.5% |
| **API Response Time** | <100ms |
| **Models Compared** | 4 classification algorithms |

### **Business Impact**
- ✅ Catches **40 out of 44** heart disease patients
- ✅ **27% reduction** in false alarms
- ✅ **Real-time predictions** for clinical use
- ✅ **Interpretable results** with feature importance
- ✅ **User-friendly interface** for healthcare workers

---


---




### **Data Sources**
- UCI Machine Learning Repository
- Cleveland Heart Disease Dataset
- Original dataset contributors






---
