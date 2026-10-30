from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "modelos" / "modelo_rutaia.joblib"
DATA_PATH = ROOT / "datos" / "rutas_simuladas.csv"

st.set_page_config(page_title="RutaIA", page_icon="🛣️", layout="wide")
st.title("🛣️ RutaIA")
st.subheader("Estimación académica del tiempo de recorrido")
st.info(
    "Este prototipo utiliza datos sintéticos generados para un proyecto universitario. "
    "No consulta tráfico en tiempo real y sus estimaciones no deben usarse para planificar viajes reales."
)

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo. Ejecuta primero: python entrenar_modelo.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
model_name = bundle.get("model_name", "Modelo entrenado")

with st.sidebar:
    st.header("Características de la ruta")
    distance = st.slider("Distancia (km)", 1.0, 15.0, 5.0, 0.5)
    hour = st.slider("Hora de salida", 5, 23, 7)
    traffic = st.select_slider("Nivel de tráfico", options=[1,2,3,4,5], value=3)
    lights = st.slider("Número de semáforos", 0, 30, 8)
    rain_label = st.selectbox("¿Llueve?", ["No", "Sí"])
    rain = 1 if rain_label == "Sí" else 0
    rush = 1 if ((7 <= hour <= 9) or (17 <= hour <= 19)) else 0
    calculate = st.button("Estimar tiempo", type="primary", use_container_width=True)

st.markdown("### Resultado de la estimación")
st.write("Introduce los datos en el panel lateral y pulsa **Estimar tiempo**.")
if calculate:
    row = pd.DataFrame([{
        "Distancia_km": distance,
        "Hora_salida": hour,
        "Trafico_1a5": traffic,
        "Semaforos": lights,
        "Lluvia_0o1": rain,
        "Hora_pico_0o1": rush
    }])[features]
    prediction = max(0, float(model.predict(row)[0]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Tiempo estimado", f"{prediction:.1f} min")
    col2.metric("Distancia", f"{distance:.1f} km")
    col3.metric("Modelo", model_name)
    st.caption("Resultado de una simulación académica; no representa tráfico real.")

st.divider()
st.markdown("### ¿Qué variables utiliza el modelo?")
st.write("Distancia, hora de salida, nivel de tráfico, número de semáforos, lluvia y hora pico.")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    with st.expander("Explorar la base de datos sintética"):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Base de datos: {len(df)} recorridos sintéticos.")
        st.bar_chart(df.groupby("Trafico_1a5")["Tiempo_min"].mean())

st.markdown("---")
st.caption("Proyecto académico RutaIA | Datos sintéticos | Python + scikit-learn + Streamlit")
