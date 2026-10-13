# mystreamlit.py
# Iker Alcaide Velasco, 100498961
# Alejandra Cardó González, 100499061
# PARA CORRER LA APP INTRODUCIR EN TERMINAL: streamlit run mystreamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# Configuración
st.set_page_config(
    page_title="Predicción Depósito Bancario",
    layout="wide"
)
 
# Cargar modelo
@st.cache_resource
def load_model():
    with open("modelo_final.pkl", "rb") as f:
        return cloudpickle.load(f)
 
modelo = load_model()
 
# Cabecera
st.title("Predicción de Contratación de Depósito Bancario")
st.caption("Introduce los datos del cliente para predecir si contratará un depósito a plazo fijo.")
st.divider()
 
# Formulario
st.subheader("Datos del cliente")
 
col1, col2, col3 = st.columns(3)
 
# Columna 1: Variables numéricas continuas
with col1:
    st.markdown("**Variables numéricas**")
 
    age = st.slider("Edad", min_value=18, max_value=95, value=40,
                    help="Edad del cliente en años")
 
    balance = st.slider("Balance anual (€)", min_value=-7000, max_value=82000,
                         value=0, step=100,
                         help="Balance medio anual de la cuenta")
 
    duration = st.slider("Duración última llamada (s)", min_value=0,
                          max_value=4000, value=200, step=10,
                          help="Duración en segundos del último contacto")
 
    campaign = st.slider("Contactos en esta campaña", min_value=1,
                          max_value=63, value=1,
                          help="Número de contactos realizados en la campaña actual")
 
    previous = st.slider("Contactos en campañas anteriores", min_value=0,
                          max_value=58, value=0,
                          help="Número de contactos antes de esta campaña")
 
    day = st.slider("Día del mes del contacto", min_value=1, max_value=31,
                     value=15)
 
    pdays = st.number_input(
        "Días desde último contacto previo (-1 = nunca contactado)",
        min_value=-1, max_value=871, value=-1, step=1,
        help="Días transcurridos desde el último contacto en campaña anterior"
    )
 
# Columna 2: Variables categóricas (perfil cliente)
with col2:
    st.markdown("**Perfil del cliente**")
 
    job = st.selectbox("Trabajo", [None, "admin.", "blue-collar", "entrepreneur",
                        "housemaid", "management", "retired", "self-employed",
                        "services", "student", "technician", "unemployed"],
                        index=0,
                        help="Ocupación del cliente (None = desconocido)")
 
    marital = st.selectbox("Estado civil", [None, "married", "single", "divorced"],
                            index=0,
                            help="Estado civil del cliente (None = desconocido)")
 
    education = st.selectbox("Nivel educativo",
                              ["unknown", "primary", "secondary", "tertiary"],
                              index=0,
                              help="Nivel de estudios del cliente")
 
    default = st.radio("¿Tiene crédito en default?", ["no", "yes"],
                        horizontal=True)
 
    housing = st.radio("¿Tiene hipoteca?", ["no", "yes"],
                        horizontal=True)
 
    loan = st.radio("¿Tiene préstamo personal?", ["no", "yes"],
                     horizontal=True)
 
# Columna 3: Variables de campaña
with col3:
    st.markdown("**Datos de la campaña**")
 
    contact = st.selectbox("Tipo de contacto",
                            ["unknown", "cellular", "telephone"],
                            index=0,
                            help="Canal a través del que se contactó al cliente")
 
    month = st.selectbox("Mes del último contacto",
                          ["jan", "feb", "mar", "apr", "may", "jun",
                           "jul", "aug", "sep", "oct", "nov", "dec"],
                          index=4,
                          help="Mes en que se realizó el último contacto")
 
    poutcome = st.selectbox("Resultado campaña anterior",
                             ["unknown", "success", "failure", "other"],
                             index=0,
                             help="Resultado del contacto en la campaña anterior")
 
    st.markdown("---")
    st.markdown("**Resumen de valores introducidos**")
    st.markdown(f"- Edad: **{age}** | Balance: **{balance}€**")
    st.markdown(f"- Duración: **{duration}s** | Campaña: **{campaign}** contactos")
    st.markdown(f"- Trabajo: **{job}** | Estado civil: **{marital}**")
    st.markdown(f"- Contacto previo: **{'Sí' if pdays != -1 else 'No'}**")
 
# Botón de predicción
st.divider()
_, center, _ = st.columns([1, 2, 1])
with center:
    predict_btn = st.button("Realizar Predicción", type="primary",
                             use_container_width=True)
 
# Resultado
# if predict_btn:
    input_data = pd.DataFrame([{
        "age":       age,
        "balance":   balance,
        "day":       day,
        "duration":  duration,
        "campaign":  campaign,
        "previous":  previous,
        "pdays":     pdays,
        "month":     month,
        "job":       job,       # puede ser None, pipeline imputa
        "marital":   marital,   # puede ser None, pipeline imputa
        "education": education,
        "default":   default,
        "housing":   housing,
        "loan":      loan,
        "contact":   contact,
        "poutcome":  poutcome,
    }])

    prediction = modelo.predict(input_data)[0]
    has_proba  = hasattr(modelo, "predict_proba")
    proba      = modelo.predict_proba(input_data)[0] if has_proba else None
 
    st.divider()
    st.subheader("Resultado")
 
    col_res1, col_res2 = st.columns(2)
 
    with col_res1:
        if prediction == "yes":
            st.success("El cliente SÍ contratará el depósito")
        else:
            st.error("El cliente NO contratará el depósito")
 
    with col_res2:
        if proba is not None:
            classes  = list(modelo.classes_)
            prob_yes = proba[classes.index("yes")]
            prob_no  = proba[classes.index("no")]
            c1, c2 = st.columns(2)
            c1.metric("P(contrata)", f"{prob_yes:.1%}")
            c2.metric("P(no contrata)", f"{prob_no:.1%}")
            st.progress(prob_yes, text="Probabilidad de contratación")
 
    with st.expander("Ver datos del cliente enviados al modelo"):
        st.dataframe(input_data, use_container_width=True)
 
# Footer 
st.divider()
st.caption("Práctica ML · UC3M 2025-26 · Iker Alcaide Velasco y Alejandra Cardó González")