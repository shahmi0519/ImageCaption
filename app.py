# --- Python code starts here ---
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from PIL import Image
import os

# --- Paths to your files in Colab (upload them to Colab first) ---
MODEL_PATH = "/content/final_model.h5"
TOKENIZER_PATH = "/content/tokenizer.pkl"
HISTORY_PATH = "/content/train_history.pkl"
MAX_LENGTH = 34

# --- Streamlit page ---
st.set_page_config(page_title="📸 Image Caption Generator", layout="centered")
st.title("📸 Image Caption Generator (TensorFlow + Flickr8k)")
st.markdown("Upload an image to generate an AI-powered descriptive caption!")

# --- Load caption model ---
@st.cache_resource
def load_caption_model():
    return load_model(MODEL_PATH)

# --- Load tokenizer ---
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

# --- Load feature extractor once ---
@st.cache_resource
def load_feature_extractor():
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', weights='imagenet')
    return model

# --- Load training history ---
@st.cache_resource
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# --- Extract features ---
def extract_features(img_path, feature_model):
    img = kimage.load_img(img_path, target_size=(299, 299))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = feature_model.predict(x)
    return feature

# --- Generate caption (greedy search) ---
def generate_caption(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        if yhat == 0:
            break
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.replace('<start>', '').replace('<end>', '').strip()
    return final.capitalize()

# --- Load resources ---
caption_model = load_caption_model()
tokenizer = load_tokenizer()
feature_model = load_feature_extractor()
history = load_history()

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "/content/temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    image = Image.open(temp_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🪄 Generate Caption"):
        with st.spinner("Generating caption..."):
            feature = extract_features(temp_path, feature_model)
            caption = generate_caption(caption_model, tokenizer, feature, MAX_LENGTH)
            st.success("✨ Caption Generated!")
            st.markdown(f"**Generated Caption:** _{caption}_")
    
    os.remove(temp_path)

# --- Optional: Show training history ---
if history:
    import matplotlib.pyplot as plt
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots()
    ax.plot(history.get('loss', []), label='Training Loss')
    ax.plot(history.get('val_loss', []), label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
