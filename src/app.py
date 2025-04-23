from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
import pandas as pd
import joblib  # O import pickle si usaste eso para guardar
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción Depresión (XGBoost)",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_data # Cache para no recargar en cada interacción
def load_assets():
    try:
        
        model = joblib.load('tu_modelo.joblib') 
        scaler = joblib.load('mi_scaler.joblib') 
        return model, scaler
    except FileNotFoundError:
        st.error("Error")
        return None, None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los archivos: {e}")
        return None, None

model, scaler = load_assets()


feature_order = [
    'History of Mental Illness_num', 'Chronic Medical Conditions_num',
    'History of Substance Abuse_num', 'Family History of Depression_num',
    'Sleep Patterns_num', 'Age', 'Number of Children', 'Employment Status_num'
]


sleep_mapping = {'Fair': 0, 'Good': 1, 'Poor': 2}
employment_mapping = {'Unemployed': 0, 'Employed': 1}
binary_mapping = {'No': 0, 'Yes': 1}

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧠 Predicción de Posibilidad de Depresión (Modelo XGBoost)")
st.write("""
Esta aplicación utiliza un modelo XGBoost para estimar la probabilidad 
de que una persona pueda estar experimentando síntomas depresivos, basándose en 
factores de salud e historial.
""")

#ENTRADA DE DATOS DEL USUARIO 
st.sidebar.header("Ingresa tus Datos:")

def user_input_features():
    
    age = st.sidebar.slider('Edad (Age)', 18, 100, 35) # Ajusta rango/default si es necesario
    num_children = st.sidebar.number_input('Número de Hijos (Number of Children)', min_value=0, max_value=20, value=0, step=1)
    
    # Selectores para variables binarias ('Yes'/'No')
    hist_mental = st.sidebar.selectbox('¿Historial de Enfermedad Mental? (History of Mental Illness)', ['No', 'Yes'])
    chronic_cond = st.sidebar.selectbox('¿Condiciones Médicas Crónicas? (Chronic Medical Conditions)', ['No', 'Yes'])
    hist_substance = st.sidebar.selectbox('¿Historial de Abuso de Sustancias? (History of Substance Abuse)', ['No', 'Yes'])
    family_hist = st.sidebar.selectbox('¿Historial Familiar de Depresión? (Family History of Depression)', ['No', 'Yes'])
    
    
    sleep = st.sidebar.selectbox('¿Cómo son tus Patrones de Sueño? (Sleep Patterns)', ['Fair', 'Good', 'Poor'])
    employment = st.sidebar.selectbox('¿Estado Laboral? (Employment Status)', ['Unemployed', 'Employed'])

    #Crear Diccionario con los datos
    data = {
        'Age': age,
        'Number of Children': num_children,
        'History of Mental Illness': hist_mental,
        'Chronic Medical Conditions': chronic_cond,
        'History of Substance Abuse': hist_substance,
        'Family History of Depression': family_hist,
        'Sleep Patterns': sleep,
        'Employment Status': employment
    }
    
    return data

input_data = user_input_features()

# Mostrar Datos Ingresados
st.subheader("Resumen de Datos Ingresados:")

st.write(pd.DataFrame([input_data])) # Muestra como tabla
st.markdown("---")


# --- PREDICCIÓN (solo si se cargaron los assets) ---
if model is not None and scaler is not None:
    if st.button("Realizar Predicción"):
        
        # --- PREPROCESAMIENTO DE LA ENTRADA ---
        try:
            # 1. Aplicar Mapeos
            processed_data = {}
            processed_data['History of Mental Illness_num'] = binary_mapping[input_data['History of Mental Illness']]
            processed_data['Chronic Medical Conditions_num'] = binary_mapping[input_data['Chronic Medical Conditions']]
            processed_data['History of Substance Abuse_num'] = binary_mapping[input_data['History of Substance Abuse']]
            processed_data['Family History of Depression_num'] = binary_mapping[input_data['Family History of Depression']]
            processed_data['Sleep Patterns_num'] = sleep_mapping[input_data['Sleep Patterns']]
            processed_data['Employment Status_num'] = employment_mapping[input_data['Employment Status']]
            processed_data['Age'] = input_data['Age']
            processed_data['Number of Children'] = input_data['Number of Children']

            # 2. Asegurar el Orden Correcto de las Columnas
            # Creamos una lista con los valores en el orden de feature_order
            final_input_list = [processed_data[col] for col in feature_order]
            
            final_input_array = np.array(final_input_list).reshape(1, -1)

            # 3. Aplicar StandardScaler
            scaled_input = scaler.transform(final_input_array)

            # 4. Realizar Predicción
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input) # Probabilidades

            # --- MOSTRAR RESULTADO ---
            st.subheader("Resultado de la Predicción:")
            
            # Interpretar resultado
            if prediction[0] == 1:
                st.error("El modelo predice una **ALTA probabilidad** de síntomas depresivos (Resultado: 1).", icon="🚨")
                probability_depression = prediction_proba[0][1] * 100 # Probabilidad de clase 1 (Depresión)
                st.metric(label="Probabilidad Estimada de Depresión", value=f"{probability_depression:.2f}%")
                st.info("**Recuerda:** Esto no es un diagnóstico. Considera buscar apoyo profesional.")
            else:
                st.success("El modelo predice una **BAJA probabilidad** de síntomas depresivos (Resultado: 0).", icon="✅")
                probability_no_depression = prediction_proba[0][0] * 100 # Probabilidad de clase 0 (No Depresión)
                st.metric(label="Probabilidad Estimada de NO Depresión", value=f"{probability_no_depression:.2f}%")
                st.info("Mantener hábitos saludables es importante para el bienestar general.")

            # Opcional: Mostrar probabilidades de ambas clases
            # st.write("Probabilidades detalladas:")
            # st.write(f" - Probabilidad de NO Depresión (Clase 0): {prediction_proba[0][0]:.4f}")
            # st.write(f" - Probabilidad de Depresión (Clase 1): {prediction_proba[0][1]:.4f}")

        except KeyError as e:
             st.error(f"Error en el mapeo de categorías: {e}. Verifica que las opciones seleccionadas coincidan con los mapeos internos.")
        except Exception as e:
            st.error(f"Ocurrió un error durante el preprocesamiento o la predicción: {e}")

else:
    # Mensaje si no se pudieron cargar modelo/scaler
    st.error("La aplicación no puede realizar predicciones porque faltan archivos esenciales (modelo o scaler).")


st.markdown("---")
st.caption("Descargo de responsabilidad: Esta aplicación es una herramienta demostrativa basada en un modelo XGBoost y no proporciona asesoramiento médico. Los resultados no deben considerarse un diagnóstico. Consulta siempre a un profesional de la salud calificado para cualquier preocupación sobre tu salud mental.")