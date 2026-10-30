# RutaIA

Academic prototype for estimated urban travel time. All records in `datos/rutas_simuladas.csv` are synthetic. Do not interpret predictions as validated travel advice or real-time traffic estimates.

## Files
- `generar_datos.py`: reproducibly generates 1,000 fictional trips.
- `analisis_calculo.py`: partial derivatives, gradient, Hessian and visualizations.
- `entrenar_modelo.py`: compares Linear Regression and Random Forest on an 80/20 split.
- `app.py`: Streamlit web interface.
- `datos/RutaIA_datos_y_analisis.xlsx`: data, summary, heat map and notes.
- `RutaIA_Guia_Paso_a_Paso.docx`: Spanish instructions and code overview.
- `RoutIA_Poster_English_Template_Format.pptx`: English poster using the provided institution template.

## Run
1. `python -m pip install -r requirements.txt`
2. `python generar_datos.py`
3. `python analisis_calculo.py`
4. `python entrenar_modelo.py`
5. `streamlit run app.py`

## Publish for free
Push the contents to GitHub. In Streamlit Community Cloud, select the GitHub repository, branch `main`, and `app.py`. See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy.
