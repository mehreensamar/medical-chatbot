# app.py — FastAPI Backend

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load models and data
model = joblib.load("rf_structured.pkl")
symptom_list = joblib.load("all_symptoms.pkl")
class_names = joblib.load("class_names.pkl")

# Disease list for risk-based focus
focus_diseases = ["Hypertension", "Cardiovascular Disease", "Type 2 Diabetes"]

app = FastAPI()

# Allow frontend (Streamlit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class SymptomRequest(BaseModel):
    symptoms: list[str]

@app.post("/predict")
def predict_disease(data: SymptomRequest):
    symptoms = data.symptoms
    input_vector = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    df_input = pd.DataFrame([input_vector], columns=symptom_list)

    probs = model.predict_proba(df_input)[0]
    result = []

    # Calculate risk levels for focus diseases
    for disease in focus_diseases:
        idx = class_names.index(disease)
        confidence = probs[idx]
        if confidence >= 0.7:
            risk = "High"
        elif confidence >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"
        result.append({
            "disease": disease,
            "confidence": round(confidence, 3),
            "risk": risk
        })

    # Top 3 overall diseases
    top3_indices = probs.argsort()[-3:][::-1]
    top_predictions = [{
        "disease": class_names[i],
        "confidence": round(probs[i], 3)
    } for i in top3_indices]

    return {
        "focus_diseases": result,
        "top_diseases": top_predictions
    }
