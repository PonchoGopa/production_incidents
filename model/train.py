import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "dataset.csv")

df = pd.read_csv(file_path)

p95 = df["minutes"].quantile(0.95)
df = df[df["minutes"] <= p95]

avg_minutes = df.groupby("description_id")["minutes"].mean()

df["avg_minutes_by_description"] = df["description_id"].map(avg_minutes)

avg_machine = df.groupby("machine_id")["minutes"].mean()

df["avg_minutes_by_machine"] = df["machine_id"].map(avg_machine)

# combinación máquina + descripción
avg_combo = df.groupby(["machine_id", "description_id"])["minutes"].mean()

df["avg_minutes_by_combo"] = df.set_index(
    ["machine_id", "description_id"]
).index.map(avg_combo)

avg_desc_path = os.path.join(BASE_DIR, "avg_minutes_by_description.pkl")
avg_machine_path = os.path.join(BASE_DIR, "avg_minutes_by_machine.pkl")

joblib.dump(avg_minutes.to_dict(), avg_desc_path)
joblib.dump(avg_machine.to_dict(), avg_machine_path)

print("Archivos guardados en:")
print(avg_desc_path)
print(avg_machine_path)


print("Nuevo shape después de limpiar:", df.shape)

def categorize(minutes):
    if minutes <= 30:
        return "baja"
    elif minutes <= 90:
        return "media"
    else:
        return "alta"

df["severity"] = df["minutes"].apply(categorize)



X = df.drop(columns=["minutes"], errors="ignore")
y = df["minutes"]
y_log = np.log1p(y)

# Columnas
categorical_cols = ["machine_id", "area_id", "description_id", "programmed_stop_id"]
numeric_cols = ["hour", "day_of_week", "avg_minutes_by_description", "avg_minutes_by_machine","avg_minutes_by_combo"]

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


df = df.sort_values(by=["day_of_week", "hour"])


split_index = int(len(df) * 0.8)

y_train = y_log.iloc[:split_index]   
y_test = y.iloc[split_index:]        

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

# Entrenar
pipeline.fit(X_train, y_train)

# Predicción
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, 0, None)

print("Predicciones ejemplo:", y_pred[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f} minutos")
print(f"RMSE: {rmse:.2f} minutos")

# ==============================
# DIAGNÓSTICO DEL MODELO
# ==============================

df_results = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})

def categorize(minutes):
    if minutes <= 20:
        return "baja"
    elif minutes <= 80:
        return "media"
    else:
        return "alta"

df_results["real_category"] = df_results["y_true"].apply(categorize)

# Error absoluto
df_results["error"] = abs(df_results["y_true"] - df_results["y_pred"])

print("\nError promedio por categoría real:")
print(df_results.groupby("real_category")["error"].mean())

# Distribución
print("\nDistribución REAL:")
print(pd.Series(y_test).describe())

print("\nDistribución PREDICHA:")
print(pd.Series(y_pred).describe())

# Error relativo
df_results["relative_error"] = df_results["error"] / (df_results["y_true"] + 1)

print("\nError relativo promedio:")
print(df_results["relative_error"].mean())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

joblib.dump(pipeline, model_path)

print(f"Modelo guardado en: {model_path}")

#print(f"MAE: {mae:.2f} minutos")
#print(df.describe())
#print(df.isnull().sum())