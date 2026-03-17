import streamlit as st
import numpy as np
import pickle
from gtts import gTTS
from googletrans import Translator

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Set custom web page title
st.set_page_config(page_title="Image Caption Generator", page_icon="📷")

# App title
st.title("📸 Image Caption Generator")
st.markdown("Upload an image to generate a caption, translate it, and listen to it in your preferred language!")

# Load VGG16 model (excluding final layer)
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load trained caption model
model = load_model('model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Supported languages
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Russian": "ru",
    "Bengali": "bn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Arabic": "ar"
}
selected_language = st.selectbox("🌐 Select language for caption and audio", list(language_options.keys()))
lang_code = language_options[selected_language]

# Upload image
uploaded_image = st.file_uploader("📁 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.subheader("🖼️ Uploaded Image")
    st.image(uploaded_image, use_container_width=True)

    with st.spinner("🔍 Generating caption..."):
        # Preprocess image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features
        image_features = vgg_model.predict(image, verbose=0)

        # Max length of caption
        max_caption_length = 35

        # Helper to map index to word
        def get_word_from_index(index, tokenizer):
            return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

        # Generate caption
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                if predicted_word is None:
                    break
                caption += " " + predicted_word
                if predicted_word == "endseq":
                    break
            return caption

        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

        # Translate caption
        translator = Translator()
        translated = translator.translate(generated_caption, dest=lang_code)
        translated_caption = translated.text

        # Convert translated text to speech
        tts = gTTS(translated_caption, lang=lang_code)
        audio_path = "predicted_caption.mp3"
        tts.save(audio_path)

    # Show results
    st.markdown("### 📝 Captions")
    st.write(f"**Original (English):** {generated_caption}")
    st.write(f"**Translated ({selected_language}):** {translated_caption}")

    st.markdown("### 🔊 Caption Audio")
    st.audio(audio_path, format='audio/mp3')
