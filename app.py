from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

app = FastAPI()

# Allow requests from all origins (you can restrict it later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
try:
    with open("rf_structured.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("⚠️ Model file 'rf_structured.pkl' not found.")

# Define the input data schema
class SymptomInput(BaseModel):
    symptoms: list[int]  # e.g., [0, 1, 0, ..., 1]

# Test root route
@app.get("/")
def root():
    return {"message": "Disease prediction backend is running."}

# Prediction route
@app.post("/predict")
def predict_disease(data: SymptomInput):
    if model is None:
        return {"error": "Model not loaded. Please check 'rf_structured.pkl' file."}
    try:
        input_array = np.array([data.symptoms])
        prediction = model.predict(input_array)
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
