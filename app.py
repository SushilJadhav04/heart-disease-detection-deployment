# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
import time
from datetime import datetime
# Add this after your imports
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64
import tempfile
import os
import plotly.express as px
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Detection System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
THRESHOLD = 0.37  # Default, will be updated from API

# ============================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================
def plot_feature_importance():
    """Create feature importance chart"""
    # Feature importance values from your model
    features = [
        'Max Heart Rate', 'Age', 'Cholesterol', 
        'Resting BP', 'ST Depression', 'ST Slope',
        'Thalassemia', 'Chest Pain', 'Major Vessels',
        'Resting ECG', 'Sex', 'Exercise Angina',
        'Fasting Blood Sugar'
    ]
    
    importance = [
        0.177, 0.149, 0.139,
        0.130, 0.108, 0.069,
        0.062, 0.054, 0.043,
        0.030, 0.022, 0.016,
        0.009
    ]
    
    # Create DataFrame and sort
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    df_imp = df_imp.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=df_imp['Importance'],
            y=df_imp['Feature'],
            orientation='h',
            marker=dict(
                color=df_imp['Importance'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=df_imp['Importance'].apply(lambda x: f'{x:.1%}'),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="🔍 Feature Importance (What influences prediction most?)",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )
    
    return fig

# ============================================
# RISK FACTORS ANALYSIS
# ============================================
def analyze_risk_factors(age, sex, cholesterol, resting_bp, max_hr, 
                        fasting_bs, exercise_angina, chest_pain):
    """Analyze individual risk factors"""
    risk_factors = []
    
    # Age risk
    if age > 55:
        risk_factors.append(("🔴 Age > 55", f"Your age ({age}) increases cardiovascular risk"))
    elif age > 45:
        risk_factors.append(("🟡 Age 45-55", "Moderate age-related risk"))
    
    # Cholesterol risk
    if cholesterol > 240:
        risk_factors.append(("🔴 High Cholesterol", f"{cholesterol} mg/dl (High)"))
    elif cholesterol > 200:
        risk_factors.append(("🟡 Borderline Cholesterol", f"{cholesterol} mg/dl"))
    else:
        risk_factors.append(("🟢 Normal Cholesterol", f"{cholesterol} mg/dl"))
    
    # Blood pressure risk
    if resting_bp > 140:
        risk_factors.append(("🔴 High BP", f"{resting_bp} mm Hg (Hypertension)"))
    elif resting_bp > 120:
        risk_factors.append(("🟡 Elevated BP", f"{resting_bp} mm Hg"))
    
    # Heart rate risk
    if max_hr < 100 and age < 60:
        risk_factors.append(("🔴 Low Max HR", f"{max_hr} bpm (Below normal)"))
    
    # Blood sugar risk
    if fasting_bs == 1:
        risk_factors.append(("🔴 High Fasting Blood Sugar", ">120 mg/dl"))
    
    # Exercise angina risk
    if exercise_angina == 1:
        risk_factors.append(("🔴 Exercise Induced Angina", "Chest pain during exercise"))
    
    # Chest pain type risk
    if chest_pain >= 2:
        risk_factors.append(("🔴 Significant Chest Pain", f"Type {chest_pain}"))
    
    return risk_factors

# ============================================
# MODEL COMPARISON DASHBOARD - FIXED VERSION (ONLY ONE!)
# ============================================
def show_model_comparison():
    """Display model comparison dashboard with correct values"""
    # Model performance data - CORRECT VALUES
    models = ['Random Forest', 'Logistic Regression', 'SVM', 'Decision Tree']
    
    # Correct values from your model training
    recall_pct = [97.7, 90.9, 88.6, 75.0]      # Recall percentages
    precision_pct = [56.6, 62.5, 62.9, 60.0]    # Precision percentages
    f1_scores = [0.717, 0.740, 0.735, 0.667]    # F1-Scores as decimals
    
    # Create DataFrame for display with CORRECT formatting
    df_models = pd.DataFrame({
        'Model': models,
        'Recall': [f"{r}%" for r in recall_pct],
        'Precision': [f"{p}%" for p in precision_pct],
        'F1-Score': [f"{f:.3f}" for f in f1_scores]
    })
    
    # Create chart using plotly.graph_objects
    fig = go.Figure()
    
    # Add Recall bars
    fig.add_trace(go.Bar(
        name='Recall',
        x=models,
        y=recall_pct,
        text=[f"{r}%" for r in recall_pct],
        textposition='outside',
        marker_color='#ff6b6b',
        textfont=dict(size=12)
    ))
    
    # Add Precision bars
    fig.add_trace(go.Bar(
        name='Precision',
        x=models,
        y=precision_pct,
        text=[f"{p}%" for p in precision_pct],
        textposition='outside',
        marker_color='#4ecdc4',
        textfont=dict(size=12)
    ))
    
    # Add F1-Score bars (multiply by 100 for consistent scale)
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=models,
        y=[f * 100 for f in f1_scores],  # Convert to percentage for display
        text=[f"{f:.3f}" for f in f1_scores],
        textposition='outside',
        marker_color='#45b7d1',
        textfont=dict(size=12)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='🤖 Model Performance Comparison',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Model',
            tickfont=dict(size=12),
            tickangle=0
        ),
        yaxis=dict(
            title='Score (%)',
            range=[0, 105],
            tickmode='linear',
            tick0=0,
            dtick=10,
            ticksuffix='%',
            gridcolor='lightgray',
            gridwidth=1
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig, df_models

# Custom CSS with improved styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Prediction box styling */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Risk level backgrounds */
    .high-risk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 8px solid #f44336;
    }
    
    .moderate-risk {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 8px solid #ff9800;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 8px solid #4caf50;
    }
    
    /* Confidence badges */
    .confidence-high {
        color: #f44336;
        font-weight: bold;
        padding: 8px 16px;
        background: rgba(244, 67, 54, 0.1);
        border-radius: 30px;
        display: inline-block;
        margin: 10px 0;
        border: 1px solid #f44336;
    }
    
    .confidence-moderate {
        color: #ff9800;
        font-weight: bold;
        padding: 8px 16px;
        background: rgba(255, 152, 0, 0.1);
        border-radius: 30px;
        display: inline-block;
        margin: 10px 0;
        border: 1px solid #ff9800;
    }
    
    .confidence-low {
        color: #2196f3;
        font-weight: bold;
        padding: 8px 16px;
        background: rgba(33, 150, 243, 0.1);
        border-radius: 30px;
        display: inline-block;
        margin: 10px 0;
        border: 1px solid #2196f3;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        text-align: center;
        border-bottom: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        height: 3.5rem;
        font-size: 1.2rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info and warning boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 8px solid #1976d2;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 8px solid #ff9800;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.1);
    }
    
    /* Feature importance styling */
    .feature-importance {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-left: 10px;
    }
    
    .status-online {
        background: #4caf50;
        color: white;
    }
    
    .status-offline {
        background: #f44336;
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHECK API CONNECTION AND GET MODEL INFO
# ============================================
@st.cache_resource
def check_api_connection():
    """Check if API is running and get model info"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        return None

@st.cache_resource
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        return None

# Check API status
api_status = check_api_connection()
model_info = get_model_info()

if api_status and api_status.get('model_loaded'):
    THRESHOLD = api_status.get('threshold', 0.37)
    st.sidebar.markdown(f"<span class='status-badge status-online'>✅ API Connected</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<span class='status-badge status-offline'>❌ API Offline</span>", unsafe_allow_html=True)
    st.sidebar.error("⚠️ FastAPI backend not running! Start it with: python src/api.py")

# Header
st.markdown("""
<div class='main-header'>
    <h1>❤️ Heart Disease Detection System</h1>
    <p style='font-size: 1.2rem; opacity: 0.9;'>Advanced ML-powered screening tool with optimized threshold (0.37)</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Patient Information Form")
    st.markdown("Enter all 13 clinical parameters below:")
    
    with st.form("prediction_form"):
        # Create two columns within the form
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            st.markdown("**🫀 Basic Information**")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=58, 
                                 help="Patient's age in years")
            
            sex = st.selectbox("Sex", options=[0, 1], 
                              format_func=lambda x: "👩 Female" if x == 0 else "👨 Male",
                              help="Biological sex")
            
            chest_pain = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                                     format_func=lambda x: ["0: Typical Angina", 
                                                           "1: Atypical Angina", 
                                                           "2: Non-anginal Pain", 
                                                           "3: Asymptomatic"][x],
                                     help="Type of chest pain experienced")
            
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 
                                        min_value=80, max_value=200, value=134,
                                        help="Resting blood pressure in mm Hg")
            
            cholesterol = st.number_input("Cholesterol (mg/dl)", 
                                         min_value=100, max_value=600, value=246,
                                         help="Serum cholesterol in mg/dl")
            
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                                     format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
                                     help="Fasting blood sugar > 120 mg/dl")
            
        with form_col2:
            st.markdown("**📊 Clinical Measurements**")
            resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                      format_func=lambda x: ["0: Normal", 
                                                            "1: ST-T Wave Abnormality", 
                                                            "2: Left Ventricular Hypertrophy"][x],
                                      help="Resting electrocardiographic results")
            
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=155,
                                    help="Maximum heart rate achieved")
            
            exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                          format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
                                          help="Exercise induced angina")
            
            st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, 
                                           value=0.4, step=0.1,
                                           help="ST depression induced by exercise")
            
            st_slope = st.selectbox("ST Slope", options=[0, 1, 2],
                                   format_func=lambda x: ["0: Upsloping", 
                                                         "1: Flat", 
                                                         "2: Downsloping"][x],
                                   help="Slope of peak exercise ST segment")
            
            num_vessels = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3],
                                      help="Number of major vessels colored by fluoroscopy")
            
            thalassemia = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                      format_func=lambda x: ["0: Normal", 
                                                            "1: Fixed Defect", 
                                                            "2: Reversible Defect", 
                                                            "3: Other"][x],
                                      help="Thalassemia type")
        
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

with col2:
    st.markdown("### 📊 Prediction Results")
    
    # Placeholder for results
    result_placeholder = st.empty()
    
    if submitted:
        if not api_status:
            st.error("❌ Cannot connect to API. Please start the FastAPI server first.")
            st.code("python src/api.py", language="bash")
        else:
            with st.spinner("🔄 Analyzing patient data with optimized model..."):
                # Prepare data for API
                patient_data = {
                    "age": age,
                    "sex": sex,
                    "chest_pain_type": chest_pain,
                    "resting_blood_pressure": resting_bp,
                    "cholesterol": cholesterol,
                    "fasting_blood_sugar": fasting_bs,
                    "resting_ecg": resting_ecg,
                    "max_heart_rate": max_hr,
                    "exercise_induced_angina": exercise_angina,
                    "st_depression": st_depression,
                    "st_slope": st_slope,
                    "num_major_vessels": num_vessels,
                    "thalassemia": thalassemia
                }
                
                try:
                    # Call API
                    response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results with animation
                        with result_placeholder.container():
                            prob = result['probability'] * 100
                            
                            # RISK LEVEL DETERMINATION WITH CONFIDENCE
                            if prob >= 70:
                                risk_class = "high-risk"
                                risk_title = "### 🔴 HIGH RISK DETECTED"
                                confidence_class = "confidence-high"
                                confidence_level = "High Confidence"
                            elif prob >= 50:
                                risk_class = "moderate-risk"
                                risk_title = "### 🟡 MODERATE RISK"
                                confidence_class = "confidence-moderate"
                                confidence_level = "Moderate Confidence"
                            elif prob >= 30:
                                risk_class = "moderate-risk"
                                risk_title = "### 🟠 BORDERLINE - SCREENING ALERT"
                                confidence_class = "confidence-low"
                                confidence_level = "Low Confidence - Screening Alert"
                            else:
                                risk_class = "low-risk"
                                risk_title = "### ✅ LOW RISK"
                                confidence_class = "confidence-high"
                                confidence_level = "High Confidence"
                            
                            # Display prediction box
                            st.markdown(f"<div class='prediction-box {risk_class}'>", unsafe_allow_html=True)
                            st.markdown(risk_title, unsafe_allow_html=True)
                            
                            # Large probability display
                            st.markdown(f"<h2 style='font-size: 3rem; margin: 0;'>{prob:.1f}%</h2>", unsafe_allow_html=True)
                            st.markdown("Probability of Heart Disease")
                            
                            # Progress bar
                            st.progress(result['probability'])
                            
                            # Confidence level badge
                            st.markdown(f"<span class='{confidence_class}'>🔔 {confidence_level}</span>", unsafe_allow_html=True)
                            
                            # Model threshold info
                            st.caption(f"⚙️ Model threshold: {THRESHOLD:.2f} (optimized for 90.9% recall)")
                            
                            # Message from API
                            st.markdown(f"**{result['message']}**")
                            
                            # Recommended action
                            st.markdown("---")
                            st.markdown(f"### 📋 Recommended Action")
                            
                            # Generate recommendation based on probability
                            if prob >= 70:
                                recommendation = "🚨 URGENT: Consult cardiologist within 24-48 hours"
                            elif prob >= 50:
                                recommendation = "📅 Schedule appointment with cardiologist within 2 weeks"
                            elif prob >= 30:
                                recommendation = "🔍 Schedule follow-up screening in 3 months and monitor symptoms"
                            else:
                                recommendation = "✅ Continue regular check-ups and maintain healthy lifestyle"
                            
                            st.info(recommendation)
                            
                            # Health recommendations based on prediction
                            st.markdown("#### 🏥 Health Recommendations:")
                            
                            if result['prediction'] == 1:
                                if prob >= 70:
                                    st.markdown("""
                                    - 🚨 **URGENT:** Consult cardiologist immediately
                                    - 💊 Follow prescribed medication strictly
                                    - 🥗 Adopt DASH or Mediterranean diet
                                    - 🚫 Quit smoking if applicable
                                    - 🏃 Light exercise only as approved by doctor
                                    """)
                                elif prob >= 50:
                                    st.markdown("""
                                    - 📅 Schedule appointment within 2 weeks
                                    - 💊 Discuss medication options with doctor
                                    - 🥗 Reduce salt and saturated fat intake
                                    - 🚫 Limit alcohol consumption
                                    - 🏃 Begin light walking (30 mins/day)
                                    """)
                                else:
                                    st.markdown("""
                                    - 🔍 This is a screening alert only
                                    - 📊 Monitor blood pressure at home
                                    - 📝 Keep a symptom diary
                                    - 🥗 Focus on heart-healthy diet
                                    - 🏃 Regular moderate exercise
                                    """)
                            else:
                                st.markdown("""
                                - ✅ Continue current healthy habits
                                - 🥗 Maintain balanced diet rich in fruits/vegetables
                                - 🏃 Regular exercise (150 mins/week)
                                - 🔍 Annual check-ups recommended
                                - 📊 Monitor BP and cholesterol regularly
                                """)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # UNDERSTANDING YOUR RESULT
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown("### 📊 Understanding Your Result")
                            
                            if prob >= 70:
                                st.markdown(f"""
                                **High confidence prediction:** Multiple risk factors indicate significant probability of heart disease.
                                
                                *Your result ({prob:.1f}%) is well above the model threshold ({THRESHOLD:.2f}). 
                                The model has 90.9% sensitivity, meaning it catches most actual cases.*
                                """)
                            elif prob >= 50:
                                st.markdown(f"""
                                **Moderate confidence prediction:** Some risk factors present.
                                
                                *Your result ({prob:.1f}%) is above the model threshold ({THRESHOLD:.2f}). 
                                Additional tests may be needed for confirmation.*
                                """)
                            elif prob >= 30:
                                st.markdown(f"""
                                **Low confidence screening alert:** Model is sensitive and flags borderline cases.
                                
                                *Your result ({prob:.1f}%) is just above the screening threshold ({THRESHOLD:.2f}). 
                                This is a precautionary alert, not a diagnosis.*
                                """)
                            else:
                                st.markdown(f"""
                                **Low risk prediction:** Your profile shows minimal risk factors.
                                
                                *Your result ({prob:.1f}%) is below the screening threshold ({THRESHOLD:.2f}). 
                                Continue healthy lifestyle and regular check-ups.*
                                """)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # ============================================
                            # FEATURE IMPORTANCE
                            # ============================================
                            st.markdown("---")
                            st.markdown("### 🔍 What Influenced This Prediction?")
                            imp_fig = plot_feature_importance()
                            st.plotly_chart(imp_fig, use_container_width=True)
                            
                            # ============================================
                            # RISK FACTORS ANALYSIS
                            # ============================================
                            st.markdown("---")
                            st.markdown("### ⚠️ Your Risk Factors Analysis")
                            
                            risk_factors = analyze_risk_factors(
                                age, sex, cholesterol, resting_bp, max_hr,
                                fasting_bs, exercise_angina, chest_pain
                            )
                            
                            if risk_factors:
                                # Create a grid layout for risk factors (2 columns)
                                risk_cols = st.columns(2)
                                for i, (icon_text, description) in enumerate(risk_factors):
                                    with risk_cols[i % 2]:
                                        st.markdown(f"**{icon_text}**")
                                        st.markdown(description)
                                        st.markdown("---")
                            else:
                                st.success("✅ No major risk factors detected!")
                            
                            # DISCLAIMER
                            st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                            st.markdown("### ⚠️ Important Medical Disclaimer")
                            st.markdown(f"""
                            This tool is for **screening purposes only** and has been optimized for high sensitivity 
                            (catching **90.9%** of disease cases at threshold {THRESHOLD:.2f}). 
                            This means it may flag some healthy individuals as a precaution.
                            
                            **Model Statistics:**
                            - ✅ **Recall:** 90.9% (catches 40/44 disease cases)
                            - ⚠️ **Precision:** 62.5% (some false positives expected)
                            - 🎯 **Optimized Threshold:** {THRESHOLD:.2f}
                            
                            **Always consult a qualified healthcare provider** for proper diagnosis and treatment.
                            """)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Key metrics in a grid
                            st.markdown("### 📈 Key Health Metrics")
                            metric_cols = st.columns(3)
                            
                            with metric_cols[0]:
                                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                                st.metric("Age", f"{age} years")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with metric_cols[1]:
                                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                                cholesterol_status = "⚠️ High" if cholesterol > 200 else "✅ Normal"
                                st.metric("Cholesterol", f"{cholesterol} mg/dl", 
                                         delta=cholesterol_status, 
                                         delta_color="off")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with metric_cols[2]:
                                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                                hr_status = "✅ Normal" if 60 <= max_hr <= 100 else "⚠️ Check"
                                st.metric("Max Heart Rate", f"{max_hr} bpm",
                                         delta=hr_status,
                                         delta_color="off")
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.error(response.text if response.text else "Unknown error")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure FastAPI server is running on port 8000!")
                    st.code("python src/api.py", language="bash")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================
# SIDEBAR - CORRECTED VERSION
# ============================================
with st.sidebar:
    st.markdown("### ℹ️ About This System")
    st.markdown("""
    This **Machine Learning system** predicts heart disease risk using 13 clinical parameters 
    from the UCI Heart Disease dataset with an optimized threshold for maximum safety.
    """)
    
    st.markdown("---")
    
    # OPTIMIZED MODEL PERFORMANCE
    st.markdown("### 🎯 **Optimized Model Performance**")
    
    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        st.markdown("**Recall (Sensitivity)**")
        st.markdown("<h2 style='color: #4caf50; margin:0;'>90.9%</h2>", unsafe_allow_html=True)
        st.caption("Catches 40/44 cases")
    
    with col_perf2:
        st.markdown("**Precision**")
        st.markdown("<h2 style='color: #ff9800; margin:0;'>62.5%</h2>", unsafe_allow_html=True)
        st.caption("Acceptable false positives")
    
    st.markdown(f"**Optimized Threshold:** `{THRESHOLD}`")
    st.markdown("**F1-Score:** `0.741`")
    st.markdown("**Missed Patients:** `4` (vs 11 at default)")
    st.markdown("**False Alarms:** `24` (vs 33 at 0.30)")
    
    st.markdown("---")
    
    # MODEL COMPARISON - CALL THE FUNCTION HERE!
    model_fig, model_df = show_model_comparison()
    
    # Display chart and table
    st.plotly_chart(model_fig, use_container_width=True)
    
    
    st.markdown("---")
    
    # RISK LEVEL GUIDE
    st.markdown("### 📋 **Risk Level Guide**")
    
    risk_guide = pd.DataFrame({
        'Probability': ['>70%', '50-70%', '30-50%', '<30%'],
        'Risk Level': ['🔴 High', '🟡 Moderate', '🟠 Borderline', '✅ Low'],
        'Action': ['Immediate', '2 weeks', 'Monitor', 'Routine']
    })
    st.dataframe(risk_guide, use_container_width=True)
    
    st.markdown("---")
    
    # DATASET STATISTICS
    st.markdown("### 📚 **Dataset Statistics**")
    st.markdown("""
    - **Total Records:** 400 patients
    - **With Disease:** 165 (41.25%)
    - **Without Disease:** 235 (58.75%)
    - **Source:** UCI Heart Disease Dataset
    """)
    
    st.markdown("---")
    
    # RESOURCES
    st.markdown("### 📚 **Resources**")
    st.markdown("""
    - [WHO: Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
    - [American Heart Association](https://www.heart.org/)
    - [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
    """)
    
    st.markdown("---")
    
    # FINAL DISCLAIMER
    st.warning("""
    **⚠️ IMPORTANT DISCLAIMER**
    
    This is a **SCREENING TOOL ONLY** optimized for 90.9% sensitivity. 
    
    It may flag healthy individuals as a precaution. Always consult a qualified healthcare provider for proper diagnosis.
    """)

# Footer
st.markdown("""
<div class='footer'>
    <p>❤️ Heart Disease Detection System v2.0 | Optimized Threshold: 0.37 | Recall: 90.9%</p>
    <p>Developed for Capstone Project </p>
    <p style='font-size: 0.8rem; color: #999;'>© 2026 | All rights reserved | For educational and screening purposes only</p>
</div>
""", unsafe_allow_html=True)