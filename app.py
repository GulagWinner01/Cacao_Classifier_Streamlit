import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# --- ¡ESTA LÍNEA DEBE SER LA PRIMERA DESPUÉS DE LAS IMPORTACIONES! ---
st.set_page_config(page_title="Clasificador de Cacao", layout="wide")

# --- 1. CONFIGURACIÓN INICIAL ---
MODEL_PATH = './models/best_cacao_classifier_tf_checkpoint.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ['danada', 'madura', 'verde']
UMBRAL_NEGRO_LOGICA = 25  # % de oscuridad para forzar 'danada'

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_my_model(model_path):
    print(f"🔄 Intentando cargar el modelo desde: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"❌ El archivo del modelo NO EXISTE en: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Modelo cargado correctamente.")
        print("✅ Modelo cargado con éxito.")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        print(f"❌ Error cargando el modelo: {e}")
        return None

model_tf = load_my_model(MODEL_PATH)

# --- 3. FUNCIONES DE UTILIDAD ---

def calcular_porcentaje_negro(img_rgb, umbral_oscuro=60):
    print("🧮 Calculando porcentaje de oscuridad...")
    if img_rgb.ndim == 4:
        img_rgb = img_rgb[0]
    if img_rgb.max() <= 1.0:
        img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb_uint8 = img_rgb.astype(np.uint8)

    try:
        gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        oscuro = gray < umbral_oscuro
        porcentaje_negro = np.sum(oscuro) / oscuro.size * 100
        print(f"🖤 Porcentaje de oscuridad: {porcentaje_negro:.2f}%")
        return porcentaje_negro
    except Exception as e:
        st.error(f"❌ Error en calcular_porcentaje_negro: {e}")
        print(f"❌ Error en calcular_porcentaje_negro: {e}")
        return 0.0

def preprocess_for_model(image_rgb):
    print("🧼 Preprocesando imagen...")
    image = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# --- 4. CLASIFICACIÓN Y GRAD-CAM ---
def classify_and_analyze_with_gradcam_tf(img_rgb_original, model_tf, class_names, img_height, img_width, umbral_negro_logica):
    results = {}
    figures = []

    try:
        input_tensor = preprocess_for_model(img_rgb_original)
    except Exception as e:
        st.error(f"❌ Error al preprocesar la imagen: {e}")
        print(f"❌ Error al preprocesar la imagen: {e}")
        return results, figures

    porcentaje_negro = calcular_porcentaje_negro(img_rgb_original)
    results['porcentaje_negro'] = porcentaje_negro

    try:
        preds = model_tf.predict(input_tensor, verbose=0)[0]
        pred_class_id = np.argmax(preds)
        pred_class_name = class_names[pred_class_id]
        confidence = np.max(preds)

        print(f"🔮 Predicción CNN: {pred_class_name}, Confianza: {confidence:.2f}")

        results['cnn_predicted_class_name'] = pred_class_name
        results['cnn_confidence'] = confidence

        if 'danada' in class_names and pred_class_name != 'danada' and porcentaje_negro > umbral_negro_logica:
            results['final_prediction_name'] = 'danada (forzada por oscuridad)'
            results['rule_applied'] = True
            print("⚠️ Se aplicó lógica de negocio para clasificar como 'danada'")
        else:
            results['final_prediction_name'] = pred_class_name
            results['rule_applied'] = False
    except Exception as e:
        st.error(f"❌ Error en predicción o regla: {e}")
        print(f"❌ Error en predicción o regla: {e}")
        return results, figures

    fig1 = plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb_original)
    plt.title(f"Clasificación Final: {results['final_prediction_name']}\n(CNN: {pred_class_name}, Oscuridad: {porcentaje_negro:.2f}%)")
    plt.axis('off')
    figures.append(fig1)

    try:
        saliency = Saliency(model_tf, model_modifier=ReplaceToLinear(), clone=True)
        saliency_map = saliency(score=CategoricalScore([pred_class_id]), seed_input=input_tensor)
        saliency_map = saliency_map[0].squeeze()

        gradcam = Gradcam(model_tf, model_modifier=ReplaceToLinear(), clone=True)
        cam = gradcam(score=CategoricalScore([pred_class_id]), seed_input=input_tensor, penultimate_layer=-1)
        cam = cam[0].squeeze()

        print("📸 Mapas de interpretabilidad generados correctamente.")

        fig2 = plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb_original)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_rgb_original)
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        plt.title(f"Saliency Map ({pred_class_name})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img_rgb_original)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f"Grad-CAM ({pred_class_name})")
        plt.axis('off')

        plt.tight_layout()
        figures.append(fig2)

    except Exception as e:
        st.error(f"❌ Error en Grad-CAM/Saliency: {e}")
        print(f"❌ Error en Grad-CAM/Saliency: {e}")

    return results, figures

# --- 5. UI DE STREAMLIT ---
st.title("Clasificador De Mazorcas De Cacao")
st.write("Sube una imagen de una mazorca de cacao para clasificarlo y ver el análisis de interpretabilidad.")

if model_tf is None:
    st.warning("⚠️ El modelo no se ha cargado correctamente.")
else:
    uploaded_file = st.file_uploader("📤 Sube una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        print("📂 Imagen recibida, procesando...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("📸 Imagen Subida:")
        st.image(img_rgb_original, use_column_width=True)

        st.subheader("🔍 Resultados del Análisis:")
        with st.spinner("Procesando imagen..."):
            results, figures = classify_and_analyze_with_gradcam_tf(
                img_rgb_original=img_rgb_original,
                model_tf=model_tf,
                class_names=CLASS_NAMES,
                img_height=IMG_HEIGHT,
                img_width=IMG_WIDTH,
                umbral_negro_logica=UMBRAL_NEGRO_LOGICA
            )

        if results:
            st.info(f"🧠 Predicción CNN: '{results['cnn_predicted_class_name']}' (confianza: {results['cnn_confidence']:.2f})")
            st.info(f"🖤 Porcentaje de oscuridad: {results['porcentaje_negro']:.2f}%")

            if results.get('rule_applied', False):
                st.warning(f"⚠️ Clasificación forzada a **{results['final_prediction_name']}** por lógica de negocio.")
            else:
                st.success(f"✅ Clasificación Final: {results['final_prediction_name']}")

            st.subheader("📊 Mapas de Interpretabilidad:")
            for fig in figures:
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.error("❌ No se pudieron obtener resultados.")
