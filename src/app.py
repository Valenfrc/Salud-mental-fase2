# from utils import db_connect 
# engine = db_connect()

import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import os

#CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Contigo IA",
    page_icon="🧠",
    layout="wide", # LAYOUT ANCHO
    initial_sidebar_state="expanded",
)

#Cargar modelo
@st.cache_data 
def load_assets():
    try:
   
        # Obtener la ruta del directorio donde está este script (app.py)
        script_dir = os.path.dirname(__file__) 


        model_path = os.path.join(script_dir, 'tu_modelo.joblib')
        scaler_path = os.path.join(script_dir, 'mi_scaler.joblib')

        # Usar las rutas completas para cargar
        model = joblib.load(model_path) 
        scaler = joblib.load(scaler_path) 


        return model, scaler
    except FileNotFoundError:
        # Puedes poner un mensaje más específico si quieres
        st.error("Error: No se encontraron los archivos 'tu_modelo.joblib' o 'mi_scaler.joblib' en el directorio del script.") 
        return None, None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los archivos: {e}")
        return None, None

model, scaler = load_assets()

# Orden EXACTO de columnas esperado por el scaler y el modelo
feature_order = [
    'History of Mental Illness_num', 'Chronic Medical Conditions_num',
    'History of Substance Abuse_num', 'Family History of Depression_num',
    'Sleep Patterns_num', 'Age', 'Number of Children', 'Employment Status_num'
]


sleep_mapping = {'Fair': 0, 'Good': 1, 'Poor': 2} 
employment_mapping = {'Unemployed': 0, 'Employed': 1} 
binary_mapping = {'No': 0, 'Sí': 1} 


st.title("🧠 Contigo IA")
st.write("""
Esta aplicación predice la probabilidad de que una persona pueda estar experimentando
síntomas depresivos, basándose en factores de salud e historial.
""")
st.markdown("---")

#ENTRADA DE DATOS DEL USUARIO
st.sidebar.header("📝 Ingresa tus Datos")

def user_input_features():
    
    st.sidebar.subheader("Información Personal")
    age = st.sidebar.slider('Edad', 18, 100, 35) 
    num_children = st.sidebar.number_input('Número de Hijos', min_value=0, max_value=20, value=0, step=1)
    employment = st.sidebar.selectbox('Estado Laboral', ['Unemployed', 'Employed'])
    
    st.sidebar.divider() # Separador visual
    
    st.sidebar.subheader("Historial Médico y Familiar")
    hist_mental = st.sidebar.selectbox('¿Historial de Enfermedad Mental?', ['No', 'Sí']) 
    chronic_cond = st.sidebar.selectbox('¿Condiciones Médicas Crónicas?', ['No', 'Sí']) 
    hist_substance = st.sidebar.selectbox('¿Historial de Abuso de Sustancias?', ['No', 'Sí']) 
    family_hist = st.sidebar.selectbox('¿Historial Familiar de Depresión?', ['No', 'Sí']) 
    
    st.sidebar.divider() # Separador visual
    
    st.sidebar.subheader("Hábitos")
    sleep = st.sidebar.selectbox('Patrones de Sueño', ['Fair', 'Good', 'Poor']) 
    

    data = {
        'Age': age,
        'Number of Children': num_children,
        'Historial de Enfermedad Mental': hist_mental, 
        'Condiciones Médicas Crónicas': chronic_cond,
        'Historial de Abuso de Sustancias': hist_substance,
        'Historial Familiar de Depresión': family_hist,
        'Sleep Patterns': sleep, 
        'Employment Status': employment 
    }
    
    return data

input_data = user_input_features()

#CONTENIDO PRINCIPAL (Resumen y Predicción en Columnas)


col1, col2 = st.columns(2)

with col1:
    with st.expander("Ver Resumen de Datos Ingresados"):
        display_data = {
            'Edad': input_data['Age'],
            'Núm. Hijos': input_data['Number of Children'],
            'Hist. Enf. Mental': input_data['Historial de Enfermedad Mental'],
            'Cond. Méd. Crónicas': input_data['Condiciones Médicas Crónicas'],
            'Hist. Abuso Sust.': input_data['Historial de Abuso de Sustancias'],
            'Hist. Fam. Depresión': input_data['Historial Familiar de Depresión'],
            'Patrones Sueño': input_data['Sleep Patterns'],
            'Estado Laboral': input_data['Employment Status']
        }
        st.write(pd.DataFrame([display_data])) 
        st.caption("Datos seleccionados en la barra lateral.")

with col2:
    st.subheader("📊 Resultado de la Predicción")
    
    # Botón predicción
    predict_button = st.button("Realizar Predicción", type="primary", use_container_width=True)
    
    #PREDICCIÓN
    if predict_button and model is not None and scaler is not None:
        
      
        try:
   
            processed_data = {}
            processed_data['History of Mental Illness_num'] = binary_mapping[input_data['Historial de Enfermedad Mental']] 
            processed_data['Chronic Medical Conditions_num'] = binary_mapping[input_data['Condiciones Médicas Crónicas']]
            processed_data['History of Substance Abuse_num'] = binary_mapping[input_data['Historial de Abuso de Sustancias']]
            processed_data['Family History of Depression_num'] = binary_mapping[input_data['Historial Familiar de Depresión']]
            processed_data['Sleep Patterns_num'] = sleep_mapping[input_data['Sleep Patterns']]
            processed_data['Employment Status_num'] = employment_mapping[input_data['Employment Status']]
            processed_data['Age'] = input_data['Age']
            processed_data['Number of Children'] = input_data['Number of Children']

       
            final_input_list = [processed_data[col] for col in feature_order]
            final_input_array = np.array(final_input_list).reshape(1, -1)

         
            scaled_input = scaler.transform(final_input_array)

           
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input) 

            #MOSTRAR RESULTADO (Dentro de la columna 2)
            st.markdown("---") # Separador antes del resultado
            
            if prediction[0] == 1:
                st.error("Predicción: **ALTA probabilidad** de síntomas depresivos (Resultado: 1).", icon="🚨")
                probability_depression = prediction_proba[0][1] * 100 
                st.metric(label="Probabilidad Estimada (Depresión)", value=f"{probability_depression:.1f}%")
                st.info("Considera buscar apoyo profesional.", icon="ℹ️")
            else:
                st.success("Predicción: **BAJA probabilidad** de síntomas depresivos (Resultado: 0).", icon="✅")
                probability_no_depression = prediction_proba[0][0] * 100 
                st.metric(label="Probabilidad Estimada (NO Depresión)", value=f"{probability_no_depression:.1f}%")
                st.info("Mantener hábitos saludables es importante para el bienestar general.", icon="ℹ️")

        except KeyError as e:
             st.error(f"Error en el mapeo de categorías: La opción '{e}' no se pudo convertir.", icon="⚙️")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}", icon="⚙️")

    elif predict_button and (model is None or scaler is None):
        # Mensaje si no se pudieron cargar modelo/scaler al presionar el botón
        st.error("La aplicación no puede realizar predicciones porque faltan archivos esenciales (modelo o scaler).")
    else:
        # Mensaje inicial antes de presionar el botón
        st.info("Completa los datos en la barra lateral y presiona 'Realizar Predicción'.")



st.markdown("---")
