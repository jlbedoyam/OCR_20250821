import streamlit as st
import easyocr
from PIL import Image
from groq import Groq

# =======================================
# Configuración de la página
# =======================================
st.set_page_config(page_title="OCR + LLM con Groq", page_icon="📖")

st.title("📖 OCR + LLM con Groq")
st.write("Sube una imagen y obtén una explicación generada por un modelo LLM de Groq.")

# =======================================
# Entrada API Key
# =======================================
api_key = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

if not api_key:
    st.info("👉 Ingresa tu API Key para continuar.")
    st.stop()

client = Groq(api_key=api_key)

# =======================================
# Subida de archivo
# =======================================
uploaded_file = st.file_uploader("📂 Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # OCR con EasyOCR
    with st.spinner("🔍 Extrayendo texto con EasyOCR..."):
        reader = easyocr.Reader(['es', 'en'])  # español e inglés
        results = reader.readtext(uploaded_file, detail=0)
        extracted_text = " ".join(results)

    st.subheader("📜 Texto extraído")
    if extracted_text.strip():
        st.write(extracted_text)
    else:
        st.warning("⚠️ No se detectó texto en la imagen.")
        st.stop()

    # Llamar al LLM de Groq
    with st.spinner("🤖 Consultando al LLM de Groq..."):
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Modelo disponible en Groq
                messages=[
                    {"role": "system", "content": "Eres un asistente que explica textos de forma clara y sencilla."},
                    {"role": "user", "content": f"Explica este texto:\n\n{extracted_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            st.error(f"❌ Error al consultar Groq: {e}")
            st.stop()

    st.subheader("🧠 Explicación del LLM")
    st.write(explanation)
