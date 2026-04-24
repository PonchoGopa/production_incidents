from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

# Cargar modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

app = FastAPI()

# Esquema de entrada
class Incident(BaseModel):
    machine_id: int
    area_id: int
    description_id: int
    programmed_stop_id: int
    hour: int
    day_of_week: int

# Endpoint
@app.post("/predict")
def predict(data: Incident):
    # Convertir a DataFrame
    input_df = pd.DataFrame([{
        "machine_id": str(data.machine_id),
        "area_id": str(data.area_id),
        "description_id": str(data.description_id),
        "programmed_stop_id": str(data.programmed_stop_id),
        "hour": data.hour,
        "day_of_week": data.day_of_week
    }])

    # IMPORTANTE: recrear features
    # (mismo nombre que en entrenamiento)
    input_df["avg_minutes_by_description"] = 0
    input_df["avg_minutes_by_machine"] = 0

    prediction = model.predict(input_df)[0]

    return {
        "severity": prediction
    }