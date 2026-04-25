import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.metrics import classification_report
import streamlit as st

st.write("Dashboard funcionando")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "dataset.csv")
model_path = os.path.join(BASE_DIR, "model.pkl")

# Cargar datos
df = pd.read_csv(data_path)

# Cargar modelo
model = joblib.load(model_path)

# Título
st.title("Análisis de Incidentes de Producción")

# --- Preprocesamiento igual que train.py ---
p95 = df["minutes"].quantile(0.95)
df = df[df["minutes"] <= p95]

def categorize(minutes):
    if minutes <= 20:
        return "baja"
    elif minutes <= 80:
        return "media"
    else:
        return "alta"

df["severity"] = df["minutes"].apply(categorize)

avg_minutes = df.groupby("description_id")["minutes"].mean()
df["avg_minutes_by_description"] = df["description_id"].map(avg_minutes)

avg_machine = df.groupby("machine_id")["minutes"].mean()
df["avg_minutes_by_machine"] = df["machine_id"].map(avg_machine)

# --- nueva feature: combinación máquina + descripción
avg_combo = df.groupby(["machine_id", "description_id"])["minutes"].mean()

df["avg_minutes_by_combo"] = df.set_index(
    ["machine_id", "description_id"]
).index.map(avg_combo)

X = df.drop(columns=["minutes", "severity"], errors="ignore")
y = df["severity"]

# Predicción
y_pred = model.predict(X)

# --- Visualizaciones ---

st.subheader("Distribución de Severidad Real")
st.bar_chart(y.value_counts())

st.subheader("Distribución de Severidad Predicha")
st.bar_chart(pd.Series(y_pred).value_counts())

st.subheader("Reporte de Clasificación")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())