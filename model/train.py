import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "dataset.csv")

df = pd.read_csv(file_path)

p95 = df["minutes"].quantile(0.95)
df = df[df["minutes"] <= p95]

avg_minutes = df.groupby("description_id")["minutes"].mean()

df["avg_minutes_by_description"] = df["description_id"].map(avg_minutes)

avg_machine = df.groupby("machine_id")["minutes"].mean()

df["avg_minutes_by_machine"] = df["machine_id"].map(avg_machine)

print("Nuevo shape después de limpiar:", df.shape)

def categorize(minutes):
    if minutes <= 20:
        return "baja"
    elif minutes <= 80:
        return "media"
    else:
        return "alta"

df["severity"] = df["minutes"].apply(categorize)

# Target (log transform)
df["target"] = np.log1p(df["minutes"])

p95 = df["minutes"].quantile(0.95)
df = df[df["minutes"] <= p95]
def categorize(minutes):
    if minutes <= 30:
        return "baja"
    elif minutes <= 90:
        return "media"
    else:
        return "alta"

df["severity"] = df["minutes"].apply(categorize)



X = df.drop(columns=["minutes", "target", "severity"], errors="ignore")
y = df["severity"]

# Columnas
categorical_cols = ["machine_id", "area_id", "description_id", "programmed_stop_id"]
numeric_cols = ["hour", "day_of_week", "avg_minutes_by_description", "avg_minutes_by_machine"]

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Modelo
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

# Pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


df = df.sort_values(by=["day_of_week", "hour"])


split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Entrenar
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)


from sklearn.metrics import classification_report

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

joblib.dump(pipeline, model_path)

print(f"Modelo guardado en: {model_path}")

#print(f"MAE: {mae:.2f} minutos")
#print(df.describe())
#print(df.isnull().sum())