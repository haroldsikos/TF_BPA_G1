import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === 1. Cargar modelo y transformadores ===
modelo = joblib.load("xgb_model.pkl")
encoder_mms = joblib.load("encoderMMS.pkl")
encoder_rs = joblib.load("encoderRS.pkl")
label_encoders = joblib.load("label_encoders.pkl")
onehe = joblib.load("oneHE.pkl")
trg_encoder = joblib.load("trg_EN.pkl")

# ✅ NUEVO: cargar valores únicos para los campos categóricos directos
valores_formulario = joblib.load("valores_formulario.pkl")

# === 2. Columnas por tipo de transformación ===
columnas_label = ['SEXO', 'CICLO_POSTULA']
columnas_rs = ['CALIF_FINAL']
columnas_mms = ['ANIO_NACIMIENTO', 'COLEGIO_ANIO_EGRESO']
columnas_onehot = ['MODALIDAD', 'ANIO_POSTULA']

# Columnas categóricas sin transformación
columnas_directas = ['NACIMIENTO_PAIS', 'DOMICILIO_DEPA', 'COLEGIO_DEPA', 'NACIMIENTO_DEPA',
                     'ESPECIALIDAD', 'DOMICILIO_PROV', 'COLEGIO_PROV', 'NACIMIENTO_PROV',
                     'DOMICILIO_DIST', 'COLEGIO_DIST', 'NACIMIENTO_DIST',
                     'COLEGIO']

# === 3. Interfaz Streamlit ===
st.title("🎓 Predicción de Ingreso Universitario")
st.markdown("Completa el formulario para predecir si un postulante ingresará a la universidad.")

# === 4. Formulario de entrada ===
sexo = st.selectbox("Sexo", ['FEMENINO', 'MASCULINO'])
ciclo = st.selectbox("Ciclo en el que postula", [1, 2])
modalidad = st.selectbox("Modalidad", ['ORDINARIO', 'INTERESADO', 'TALENTO BECA 18',
                                       'EXTRAORDINARIO 1 - DIPLOMADOS CON BACHILLERATO INTERNACIONAL',
                                       'EXTRAORDINARIO 2 – INGRESO DIRECTO CEPRE',
                                       'EXTRAORDINARIO INGRESO DIRECTO CEPRE-UNI',
                                       'EXTRAORDINARIO - TRASLADO EXTERNO',
                                       'EXTRAORDINARIO - VICTIMA DEL TERRORISMO',
                                       'INGRESO ESCOLAR NACIONAL', 'OTROS'])
anio_postula = st.selectbox("Año de postulación", [2022, 2023, 2024])
anio_nacimiento = st.number_input("Año de nacimiento", min_value=1980, max_value=2010, value=2004)
anio_egreso = st.number_input("Año de egreso del colegio", min_value=2000, max_value=2024, value=2021)
calif_final = st.slider("Calificación final estimada", 0.0, 20.0, 12.0, 0.1)

# === NUEVO: campos categóricos desde valores_formulario.pkl ===
nacimiento_pais = st.selectbox("País de nacimiento", valores_formulario["NACIMIENTO_PAIS"])
domicilio_depa = st.selectbox("Departamento de domicilio", valores_formulario["DOMICILIO_DEPA"])
colegio_depa = st.selectbox("Departamento del colegio", valores_formulario["COLEGIO_DEPA"])
nacimiento_depa = st.selectbox("Departamento de nacimiento", valores_formulario["NACIMIENTO_DEPA"])
especialidad = st.selectbox("Especialidad elegida", valores_formulario["ESPECIALIDAD"])
domicilio_prov = st.selectbox("Provincia de domicilio", valores_formulario["DOMICILIO_PROV"])
colegio_prov = st.selectbox("Provincia del colegio", valores_formulario["COLEGIO_PROV"])
nacimiento_prov = st.selectbox("Provincia de nacimiento", valores_formulario["NACIMIENTO_PROV"])
domicilio_dist = st.selectbox("Distrito de domicilio", valores_formulario["DOMICILIO_DIST"])
colegio_dist = st.selectbox("Distrito del colegio", valores_formulario["COLEGIO_DIST"])
nacimiento_dist = st.selectbox("Distrito de nacimiento", valores_formulario["NACIMIENTO_DIST"])
colegio = st.selectbox("Nombre del colegio", valores_formulario["COLEGIO"])

# === 5. Construcción del DataFrame ===
entrada = pd.DataFrame({
    'SEXO': [sexo],
    'CICLO_POSTULA': [ciclo],
    'MODALIDAD': [modalidad],
    'ANIO_POSTULA': [anio_postula],
    'ANIO_NACIMIENTO': [anio_nacimiento],
    'COLEGIO_ANIO_EGRESO': [anio_egreso],
    'CALIF_FINAL': [calif_final],
    'NACIMIENTO_PAIS': [nacimiento_pais],
    'DOMICILIO_DEPA': [domicilio_depa],
    'COLEGIO_DEPA': [colegio_depa],
    'NACIMIENTO_DEPA': [nacimiento_depa],
    'ESPECIALIDAD': [especialidad],
    'DOMICILIO_PROV': [domicilio_prov],
    'COLEGIO_PROV': [colegio_prov],
    'NACIMIENTO_PROV': [nacimiento_prov],
    'DOMICILIO_DIST': [domicilio_dist],
    'COLEGIO_DIST': [colegio_dist],
    'NACIMIENTO_DIST': [nacimiento_dist],
    'COLEGIO': [colegio]
})

# === 6. Aplicar transformaciones ===
# Label Encoding
for col in columnas_label:
    entrada[col] = label_encoders[col].transform(entrada[col])

# Escaladores
entrada[columnas_rs] = encoder_rs.transform(entrada[columnas_rs])
columnas_mms = encoder_mms.feature_names_in_.tolist()
entrada[columnas_mms] = encoder_mms.transform(entrada[columnas_mms])

# OneHot
entrada[columnas_onehot] = entrada[columnas_onehot].astype(str)
onehot_array = onehe.transform(entrada[columnas_onehot])
onehot_df = pd.DataFrame(onehot_array, columns=onehe.get_feature_names_out(columnas_onehot), index=entrada.index)

# Combinar todo
entrada_final = pd.concat([
    entrada[columnas_label + columnas_rs + columnas_mms],
    onehot_df
], axis=1)

# Asegurar columnas
for col in modelo.feature_names_in_:
    if col not in entrada_final.columns:
        entrada_final[col] = np.nan
entrada_final = entrada_final[modelo.feature_names_in_]

# === 7. Predicción ===
if st.button("Predecir ingreso"):
    try:
        pred = modelo.predict(entrada_final)[0]
        proba = modelo.predict_proba(entrada_final)[0][1]
        resultado = "ADMITIDO" if pred == 1 else "NO ADMITIDO"
        if resultado == "ADMITIDO":
            st.success(f"✅ El postulante sería **{resultado}** con una probabilidad de **{proba:.2%}**")
        else:
            st.error(f"❌ El postulante **no ingresaría** (probabilidad de ingreso: **{proba:.2%}**)")
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")
