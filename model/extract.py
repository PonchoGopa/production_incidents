import pandas as pd
import mysql.connector

# Conexión (ajusta a tu entorno)
conn = mysql.connector.connect(
    host="192.168.0.213",
    port=3306,
    user="admin",
    password="Ki23rmagqaz$",
    database="production_db"
)

query = """
SELECT
    t.machine_id,
    t.area_id,
    t.description_id,
    t.programmed_stop_id,
    t.minutes,
    HOUR(t.start_time) AS hour,
    DAYOFWEEK(t.date) AS day_of_week
FROM timeout t
WHERE t.minutes IS NOT NULL
AND t.description_id IS NOT NULL
AND t.machine_id IS NOT NULL;
"""

# Ejecutar query
df = pd.read_sql(query, conn)

# Limpieza mínima
df = df[df["minutes"] > 0]

# Guardar dataset
df.to_csv("dataset.csv", index=False)

print("Dataset generado:", df.shape)