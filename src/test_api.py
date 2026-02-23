# src/test_api.py
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Test cases
test_patients = [
    {
        "name": "High Risk Patient (should predict 1)",
        "data": {
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
    },
    {
        "name": "Low Risk Patient (should predict 0)",
        "data": {
            "age": 45,
            "sex": 0,
            "chest_pain_type": 0,
            "resting_blood_pressure": 120,
            "cholesterol": 180,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 170,
            "exercise_induced_angina": 0,
            "st_depression": 0.2,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 1
        }
    }
]

print("="*50)
print("TESTING HEART DISEASE PREDICTION API")
print("="*50)

# Test each patient
for patient in test_patients:
    print(f"\n📋 Testing: {patient['name']}")
    print("-"*40)
    
    # Make request
    response = requests.post(url, json=patient['data'])
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction: {result['prediction']}")
        print(f"📊 Probability: {result['probability']:.2%}")
        print(f"⚠️ Risk Level: {result['risk_level']}")
        print(f"💬 Message: {result['message']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

print("\n" + "="*50)
print("✅ TESTING COMPLETE")
print("="*50)