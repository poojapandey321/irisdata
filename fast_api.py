# fast_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd
import uvicorn

# Load trained model
model = joblib.load("model.joblib")  # Ensure this file is in the same directory

# Define FastAPI app
app = FastAPI(title="Iris Classifier API")

# Define request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define response schema
class PredictionResponse(BaseModel):
    predicted_class: Literal["setosa", "versicolor", "virginica"]

# Class labels
target_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "Iris classification model is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_species(features: IrisFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_class": target_names[prediction]}

if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=3000, reload=True)
