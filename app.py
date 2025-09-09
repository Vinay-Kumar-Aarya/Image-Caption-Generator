import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from googletrans import Translator
import torch
import os

# Streamlit Page Config

st.set_page_config(page_title="Image Caption Generator", page_icon="📷")
st.title("📸 Image Caption Generator")
st.markdown("Upload an image to generate a caption, translate it, and listen to it in your preferred language!")

# Load BLIP Model and Processor from Local Folder
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("./Model")
    model = BlipForConditionalGeneration.from_pretrained("./Model")
    return processor, model

processor, model = load_blip_model()

# Generate Caption Function
def generate_caption_with_blip(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Language Options
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Arabic": "ar",
    "Russian": "ru",
    "Bengali": "bn"
}
selected_language = st.selectbox("🌐 Select language for caption and audio", list(language_options.keys()))
lang_code = language_options[selected_language]

# Image Upload & Processing
uploaded_image = st.file_uploader("📁 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner("🔍 Generating caption using BLIP..."):
        # 1. Generate caption
        english_caption = generate_caption_with_blip(image)

        # 2. Translate caption
        translator = Translator()
        translated_caption = translator.translate(english_caption, dest=lang_code).text

        # 3. Text-to-speech
        audio_path = "predicted_caption.mp3"
        try:
            tts = gTTS(translated_caption, lang=lang_code)
            tts.save(audio_path)
        except ValueError:
            st.error("⚠️ Text-to-speech not supported for this language.")
            audio_path = None

    # Display Results
    st.markdown("### 📝 Captions")
    st.write(f"**Original (English):** {english_caption}")
    st.write(f"**Translated ({selected_language}):** {translated_caption}")

    if audio_path:
        st.markdown("### 🔊 Caption Audio")
        st.audio(audio_path, format='audio/mp3')

        if st.button("🗑️ Clear Audio File"):
            os.remove(audio_path)
            st.success("Audio file removed.")
