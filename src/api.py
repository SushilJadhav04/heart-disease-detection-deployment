# src/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os
from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Detection API",
    description="API for predicting heart disease using Random Forest model with optimized threshold",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; production can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Setup templates
templates = Jinja2Templates(directory="templates")

# Get the base path
base_path = os.path.dirname(os.path.dirname(__file__))

# Load models and scaler with enhanced error handling
print("="*60)
print("🚀 HEART DISEASE DETECTION API v2.0")
print("="*60)

try:
    # Try loading enhanced model first
    enhanced_model_path = os.path.join(base_path, 'models/enhanced/best_model.pkl')
    original_model_path = os.path.join(base_path, 'models/best_model.pkl')
    
    if os.path.exists(enhanced_model_path):
        model = joblib.load(enhanced_model_path)
        scaler = joblib.load(os.path.join(base_path, 'models/enhanced/scaler.pkl'))
        threshold_info = joblib.load(os.path.join(base_path, 'models/enhanced/threshold.pkl'))
        print("✅ Loaded ENHANCED model")
    else:
        model = joblib.load(original_model_path)
        scaler = joblib.load(os.path.join(base_path, 'models/scaler.pkl'))
        threshold_info = joblib.load(os.path.join(base_path, 'models/threshold.pkl'))
        print("✅ Loaded ORIGINAL model")
    
    # Get optimized threshold from threshold_info
    THRESHOLD = threshold_info.get('optimal_threshold', 0.37)
    
    # Get performance metrics if available
    RECALL_AT_THRESHOLD = threshold_info.get('recall_at_optimal', 0.909)
    PRECISION_AT_THRESHOLD = threshold_info.get('precision_at_optimal', 0.625)
    F1_AT_THRESHOLD = threshold_info.get('f1_at_optimal', 0.741)
    
    print(f"\n📊 Model Configuration:")
    print(f"   • Model Type: {type(model).__name__}")
    print(f"   • Optimized Threshold: {THRESHOLD:.2f}")
    print(f"   • Expected Recall: {RECALL_AT_THRESHOLD:.1%}")
    print(f"   • Expected Precision: {PRECISION_AT_THRESHOLD:.1%}")
    print(f"   • Expected F1-Score: {F1_AT_THRESHOLD:.3f}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️ Using fallback configuration")
    model = None
    scaler = None
    THRESHOLD = 0.37  # Fallback to optimized value
    RECALL_AT_THRESHOLD = 0.909
    PRECISION_AT_THRESHOLD = 0.625
    F1_AT_THRESHOLD = 0.741

print("="*60)

# Define the input data structure
class PatientData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_induced_angina: int
    st_depression: float
    st_slope: int
    num_major_vessels: int
    thalassemia: int

    class Config:
        schema_extra = {
            "example": {
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
        }

# Define the response structure
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    confidence_level: str
    confidence_score: float
    message: str
    recommendation: str
    model_threshold: float

@app.get("/")
async def root():
    return {
        "message": "Heart Disease Detection API",
        "status": "active",
        "version": "2.0.0",
        "threshold": THRESHOLD,
        "model_performance": {
            "recall": f"{RECALL_AT_THRESHOLD:.1%}",
            "precision": f"{PRECISION_AT_THRESHOLD:.1%}",
            "f1_score": f"{F1_AT_THRESHOLD:.3f}"
        }
    }

@app.get("/health")
async def health_check():
    if model is not None and scaler is not None:
        return {
            "status": "healthy", 
            "model_loaded": True,
            "threshold": THRESHOLD,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "unhealthy", 
            "model_loaded": False,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "threshold": THRESHOLD,
        "performance": {
            "recall": f"{RECALL_AT_THRESHOLD:.1%}",
            "precision": f"{PRECISION_AT_THRESHOLD:.1%}",
            "f1_score": f"{F1_AT_THRESHOLD:.3f}"
        },
        "features": [
            "age", "sex", "chest_pain_type", "resting_blood_pressure",
            "cholesterol", "fasting_blood_sugar", "resting_ecg",
            "max_heart_rate", "exercise_induced_angina", "st_depression",
            "st_slope", "num_major_vessels", "thalassemia"
        ]
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        # Convert input to DataFrame (handle both Pydantic v1 and v2)
        try:
            patient_dict = patient.model_dump()  # Pydantic v2
        except AttributeError:
            patient_dict = patient.dict()  # Pydantic v1
        
        input_data = pd.DataFrame([patient_dict])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction probability
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Apply optimized threshold
        prediction = int(probability >= THRESHOLD)
        
        # Determine risk level and confidence based on probability
        if probability >= 0.7:
            risk_level = "High Risk"
            confidence_level = "HIGH CONFIDENCE"
            confidence_score = 0.9
            recommendation = "🏥 Immediate consultation with cardiologist recommended"
        elif probability >= 0.5:
            risk_level = "Moderate Risk"
            confidence_level = "MODERATE CONFIDENCE"
            confidence_score = 0.7
            recommendation = "🩺 Schedule check-up within 2 weeks"
        elif probability >= 0.3:
            risk_level = "Borderline"
            confidence_level = "LOW CONFIDENCE - Screening Alert"
            confidence_score = 0.5
            recommendation = "📋 Monitor symptoms, retest in 3 months"
        else:
            risk_level = "Low Risk"
            confidence_level = "HIGH CONFIDENCE"
            confidence_score = 0.9
            recommendation = "💪 Maintain healthy lifestyle"
        
        # Create detailed message
        if prediction == 1:
            message = f"Patient shows signs of heart disease (Risk: {risk_level})"
            detailed_message = f"Based on clinical parameters, there is a {probability:.1%} probability of heart disease."
        else:
            message = f"Patient appears healthy (Risk: {risk_level})"
            detailed_message = f"Based on clinical parameters, there is a low probability ({probability:.1%}) of heart disease."
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(float(probability), 4),
            risk_level=risk_level,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            message=detailed_message,
            recommendation=recommendation,
            model_threshold=THRESHOLD
        )
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)