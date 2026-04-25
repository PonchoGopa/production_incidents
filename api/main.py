from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

# Cargar modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

avg_desc_path = os.path.join(BASE_DIR, "avg_minutes_by_description.pkl")
avg_machine_path = os.path.join(BASE_DIR, "avg_minutes_by_machine.pkl")

avg_minutes_by_description = joblib.load(avg_desc_path)
avg_minutes_by_machine = joblib.load(avg_machine_path)

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
    desc_id = str(data.description_id)
    machine_id = str(data.machine_id)

    input_df["avg_minutes_by_description"] = avg_minutes_by_description.get(desc_id, 0)
    input_df["avg_minutes_by_machine"] = avg_minutes_by_machine.get(machine_id, 0)

    prediction = model.predict(input_df)[0]

    return {
        "severity": prediction
    }