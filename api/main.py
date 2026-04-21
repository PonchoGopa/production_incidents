import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Cargar datos
df = pd.read_csv("dataset.csv")

# Target (log transform)
df["target"] = np.log1p(df["minutes"])

# Features
X = df.drop(columns=["minutes", "target"])
y = df["target"]

# Columnas
categorical_cols = ["machine_id", "area_id", "description_id", "programmed_stop_id"]
numeric_cols = ["hour", "day_of_week"]

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Modelo
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
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

# Revertir log
y_pred_real = np.expm1(y_pred)
y_test_real = np.expm1(y_test)

# Evaluación
mae = mean_absolute_error(y_test_real, y_pred_real)

print(f"MAE: {mae:.2f} minutos")