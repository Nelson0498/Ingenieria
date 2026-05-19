import streamlit as st
import xgboost as xgb
import numpy as np

# 1. Configuración de la página web
st.set_page_config(
    page_title="Predicción de Presión - Cuenca Neuquina",
    page_icon="🛢️",
    layout="centered"
)

# Título y descripción de la aplicación
st.title("🛢️ Simulador de Estimulación Hidráulica - CRISP-DM")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost Regressor** entrenado con datos públicos de la 
Secretaría de Energía para predecir la **Presión Máxima de Fractura (PSI)** en base al diseño propuesto para el pozo.
""")
st.markdown("---")

# 2. Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    model = xgb.XGBRegressor()
    # Cargamos el archivo JSON que exportamos desde Colab
    model.load_model('modelo_fracking_xgb.json')
    return model

try:
    modelo = cargar_modelo()
    st.sidebar.success("✅ Modelo XGBoost cargado con éxito")
except Exception as e:
    st.sidebar.error("❌ No se encontró el archivo 'modelo_fracking_xgb.json'")
    st.sidebar.info("Por favor, asegúrate de colocar el archivo del modelo en la misma carpeta que este script.")

# 3. Interfaz de usuario: Controles de entrada de datos en los rangos del EDA
st.subheader("🔧 Parámetros de Diseño del Pozo")

col1, col2 = st.columns(2)

with col1:
    longitud = st.slider("Longitud de Rama Horizontal (m):", min_value=100, max_value=4500, value=2200, step=50)
    cant_fracturas = st.slider("Cantidad de Etapas de Fractura:", min_value=1, max_value=100, value=45, step=1)
    potencia = st.slider("Potencia de Equipos en Superficie (HP):", min_value=5000, max_value=50000, value=25000, step=500)

with col2:
    agua = st.slider("Volumen de Agua Inyectada Total (m³):", min_value=1000, max_value=160000, value=75000, step=1000)
    arena = st.slider("Cantidad de Arena Total Bombeada (Tn):", min_value=100, max_value=15000, value=6000, step=100)

st.markdown("---")

# 4. Botón de predicción y ejecución del algoritmo
if st.button("🚀 Calcular Presión Máxima Estimada", type="primary"):
    # Organizar los datos en el mismo orden estricto que las features del entrenamiento
    # features = ['longitud_rama_horizontal_m', 'cantidad_fracturas', 'agua_inyectada_m3', 'arena_total_tn', 'potencia_equipos_fractura_hp']
    datos_entrada = np.array([[longitud, cant_fracturas, agua, arena, potencia]])
    
    # Realizar la predicción
    prediccion_psi = modelo.predict(datos_entrada)[0]
    
    # Mostrar el resultado de manera llamativa
    st.subheader("📊 Resultado de la Simulación")
    st.metric(
        label="Presión Máxima de Fractura Estimada", 
        value=f"{prediccion_psi:,.0f} PSI"
    )
    
    # Alerta técnica de seguridad basada en los límites operativos observados
    if prediccion_psi > 13000:
        st.warning("⚠️ **Alerta Operativa:** La presión estimada se encuentra cerca del límite de resistencia estándar de los casings comunes (13,500 - 15,000 PSI). Monitorear fricción.")
    else:
        st.success("✅ **Operación Segura:** La presión estimada se encuentra dentro de la ventana normal de trabajo de la Cuenca Neuquina.")

# Pie de página académico
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Trabajo Práctico - Metodología CRISP-DM aplicada a Hidrocarburos No Convencionales.")
