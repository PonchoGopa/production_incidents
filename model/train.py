import pandas as pd
import numpy as np
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

# Features
X = df.drop(columns=["minutes", "target", "severity"], errors="ignore")
y = df["severity"]

# Columnas
categorical_cols = ["machine_id", "area_id", "description_id", "programmed_stop_id"]
numeric_cols = ["hour", "day_of_week", "avg_minutes_by_description"]

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)


from sklearn.metrics import classification_report

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

#print(f"MAE: {mae:.2f} minutos")
#print(df.describe())
#print(df.isnull().sum())