import joblib
import numpy as np
import os

# Load trained heart disease model (once)
MODEL_PATH = os.path.join("models", "heart_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_heart_disease(age, bp, cholesterol, heart_rate):
    """
    Predict heart disease using trained Random Forest model.
    Returns a dict matching the format expected by result.html:
      { status, type, confidence, all_probs }
    """

    # Convert inputs to integers
    age         = int(age)
    bp          = int(bp)
    cholesterol = int(cholesterol)
    heart_rate  = int(heart_rate)

    # IMPORTANT: feature order must match training order in train_heart_model.py
    # Training used: ['age', 'trestbps', 'chol', 'thalach']
    input_data = np.array([[age, bp, cholesterol, heart_rate]])

    prediction = model.predict(input_data)[0]

    # Random Forest gives predict_proba — use it for high/low risk probability
    proba = model.predict_proba(input_data)[0]  # [prob_class0, prob_class1]

    low_risk_pct  = f"{proba[0] * 100:.1f}"
    high_risk_pct = f"{proba[1] * 100:.1f}"

    all_probs = {
        "Low Risk":  low_risk_pct,
        "High Risk": high_risk_pct
    }

    if prediction == 1:
        return {
            "status":     "High Risk of Heart Disease",
            "type":       "Cardiovascular Risk",
            "confidence": "N/A",
            "all_probs":  all_probs
        }
    else:
        return {
            "status":     "Low Risk of Heart Disease",
            "type":       "No Cardiovascular Risk",
            "confidence": "N/A",
            "all_probs":  all_probs
        }