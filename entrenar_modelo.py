from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datos" / "rutas_simuladas.csv"
MODEL_DIR = ROOT / "modelos"
OUT = ROOT / "resultados"
MODEL_DIR.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("No existe datos/rutas_simuladas.csv. Ejecuta primero generar_datos.py.")

df = pd.read_csv(DATA)
features = ["Distancia_km", "Hora_salida", "Trafico_1a5", "Semaforos", "Lluvia_0o1", "Hora_pico_0o1"]
X, y = df[features], df["Tiempo_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Regresion lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)
}
rows = []
best_name, best_model, best_mae = None, None, float("inf")
best_predictions = None

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    rows.append({"Modelo": name, "MAE_min": round(mae, 3), "RMSE_min": round(rmse, 3), "R2": round(r2, 4)})
    print(f"{name}: MAE={mae:.3f} min | RMSE={rmse:.3f} min | R²={r2:.4f}")
    if mae < best_mae:
        best_name, best_model, best_mae, best_predictions = name, model, mae, pred

pd.DataFrame(rows).to_csv(OUT / "metricas_modelo.csv", index=False, encoding="utf-8-sig")
joblib.dump({"model": best_model, "features": features, "model_name": best_name}, MODEL_DIR / "modelo_rutaia.joblib")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_predictions, alpha=0.65)
lims = [min(y_test.min(), best_predictions.min()), max(y_test.max(), best_predictions.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Tiempo simulado (min)")
plt.ylabel("Tiempo predicho (min)")
plt.title(f"Predicción vs. valor simulado — {best_name}")
plt.tight_layout()
plt.savefig(OUT / "predicciones_vs_simulados.png", dpi=160)
plt.close()

print(f"Modelo guardado: {MODEL_DIR / 'modelo_rutaia.joblib'}")
print(f"Mejor modelo según MAE: {best_name}")
