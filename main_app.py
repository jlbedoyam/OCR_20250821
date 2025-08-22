import streamlit as st
import pytesseract
from PIL import Image
import os
from groq import Groq

# =======================================
# Configuración de la API Key de Groq
# =======================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not GROQ_API_KEY:
    st.error("❌ No se encontró la API Key. Configúrala en .streamlit/secrets.toml o como variable de entorno GROQ_API_KEY.")
    st.stop()

# Inicializar cliente Groq
client = Groq(api_key=GROQ_API_KEY)

# =======================================
# Interfaz de Streamlit
# =======================================
st.title("📖 OCR + LLM con Groq")
st.write("Sube una imagen y obtén una explicación generada por un modelo LLM de Groq.")

# Subida de archivo
uploaded_file = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Extraer texto con OCR
    with st.spinner("🔍 Extrayendo texto con OCR..."):
        extracted_text = pytesseract.image_to_string(image, lang="spa")  # usa "eng" o "spa+eng" según necesidad

    st.subheader("📜 Texto extraído")
    st.write(extracted_text if extracted_text.strip() else "⚠️ No se detectó texto en la imagen.")

    if extracted_text.strip():
        # Llamar al LLM de Groq
        with st.spinner("🤖 Consultando al LLM de Groq..."):
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Modelo de Groq
                messages=[
                    {"role": "system", "content": "Eres un asistente que explica textos de forma clara y sencilla."},
                    {"role": "user", "content": f"Explica este texto:\n\n{extracted_text}"}
                ],
                temperature=0.7,
                max_tokens=500
            )

            explanation = response.choices[0].message.content

        st.subheader("🧠 Explicación del LLM")
        st.write(explanation)

