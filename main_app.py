import streamlit as st
import pytesseract
from PIL import Image
from groq import Groq

# =======================================
# Interfaz de usuario
# =======================================
st.set_page_config(page_title="OCR + LLM con Groq", page_icon="📖")

st.title("📖 OCR + LLM con Groq")
st.write("Sube una imagen con texto y obtén una explicación generada por un modelo LLM de Groq.")

# Entrada para la API Key del usuario
api_key = st.text_input("🔑 Ingresa tu API Key de Groq:", type="password")

if api_key:
    client = Groq(api_key=api_key)

    # Subida de archivo
    uploaded_file = st.file_uploader("📂 Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)

        # OCR con Tesseract
        with st.spinner("🔍 Extrayendo texto con OCR..."):
            extracted_text = pytesseract.image_to_string(image, lang="spa")  # Cambia a "eng" si es en inglés

        st.subheader("📜 Texto extraído")
        st.write(extracted_text if extracted_text.strip() else "⚠️ No se detectó texto en la imagen.")

        if extracted_text.strip():
            # Consultar al modelo LLM de Groq
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
                    explanation = None

            if explanation:
                st.subheader("🧠 Explicación del LLM")
                st.write(explanation)

else:
    st.info("👉 Ingresa tu API Key de Groq para continuar.")

