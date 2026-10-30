from pathlib import Path
import numpy as np
import pandas as pd

# Datos sintéticos para fines académicos. No representan tráfico real.
SEED = 42
N = 1000
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datos"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
dist = rng.uniform(1, 15, N).round(2)
hour = rng.integers(5, 24, N)
traffic = rng.integers(1, 6, N)
lights = np.clip(np.round(dist * rng.uniform(0.7, 2.0, N)), 0, 30).astype(int)
rain = rng.integers(0, 2, N)
rush = np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19)), 1, 0)

# Regla de simulación transparente + ruido aleatorio.
time_min = (
    2.4 * dist + 1.8 * traffic + 0.35 * lights
    + 0.12 * dist * traffic + 2 * rain + 1.2 * rush
    + rng.normal(0, 1.8, N)
)
time_min = np.maximum(time_min, 2).round(2)

df = pd.DataFrame({
    "ID": np.arange(1, N + 1),
    "Distancia_km": dist,
    "Hora_salida": hour,
    "Trafico_1a5": traffic,
    "Semaforos": lights,
    "Lluvia_0o1": rain,
    "Hora_pico_0o1": rush,
    "Tiempo_min": time_min
})
df.to_csv(OUT / "rutas_simuladas.csv", index=False, encoding="utf-8-sig")
print(f"Listo: {OUT / 'rutas_simuladas.csv'} ({len(df)} registros)")
