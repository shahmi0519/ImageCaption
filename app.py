# # --- Python code starts here ---
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as kimage
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras.models import load_model
# import numpy as np
# import pickle
# from PIL import Image
# import os


# # --- Paths to files (ensure no spaces in names!) ---
# MODEL_PATH = "final_model.h5"
# TOKENIZER_PATH = "tokenizer.pkl"
# HISTORY_PATH = "train_history.pkl"
# MAX_LENGTH = 34



# st.write("File exists?", os.path.exists(MODEL_PATH))
# st.write("File size:", os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else "N/A")


# # --- Streamlit page ---
# st.set_page_config(page_title="📸 Image Caption Generator", layout="centered")
# st.title("📸 Image Caption Generator (TensorFlow + Flickr8k)")
# st.markdown("Upload an image to generate an AI-powered descriptive caption!")

# # --- Load caption model ---
# @st.cache_resource
# def load_caption_model():
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model file not found at {MODEL_PATH}. Please check your repo.")
#         st.stop()
#     return load_model(MODEL_PATH)

# # --- Load tokenizer ---
# @st.cache_resource
# def load_tokenizer():
#     if not os.path.exists(TOKENIZER_PATH):
#         st.error(f"Tokenizer file not found at {TOKENIZER_PATH}.")
#         st.stop()
#     with open(TOKENIZER_PATH, 'rb') as f:
#         return pickle.load(f)

# # --- Load feature extractor ---
# @st.cache_resource
# def load_feature_extractor():
#     model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', weights='imagenet')
#     return model

# # --- Load training history ---
# @st.cache_resource
# def load_history():
#     if os.path.exists(HISTORY_PATH):
#         with open(HISTORY_PATH, 'rb') as f:
#             return pickle.load(f)
#     return None

# # --- Extract features ---
# def extract_features(img, feature_model):
#     img = img.resize((299, 299))
#     x = kimage.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     feature = feature_model.predict(x)
#     return feature

# # --- Generate caption ---
# def generate_caption(model, tokenizer, photo, max_length):
#     in_text = '<start>'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([photo, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         if yhat == 0:
#             break
#         word = None
#         for w, index in tokenizer.word_index.items():
#             if index == yhat:
#                 word = w
#                 break
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == '<end>':
#             break
#     final = in_text.replace('<start>', '').replace('<end>', '').strip()
#     return final.capitalize()

# # --- Load resources ---
# caption_model = load_caption_model()
# tokenizer = load_tokenizer()
# feature_model = load_feature_extractor()
# history = load_history()

# # --- File uploader ---
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("🪄 Generate Caption"):
#         with st.spinner("Generating caption..."):
#             feature = extract_features(image, feature_model)
#             caption = generate_caption(caption_model, tokenizer, feature, MAX_LENGTH)
#             st.success("✨ Caption Generated!")
#             st.markdown(f"**Generated Caption:** _{caption}_")

# # --- Optional: Show training history ---
# if history:
#     import matplotlib.pyplot as plt
#     st.subheader("📈 Training Performance")
#     fig, ax = plt.subplots()
#     ax.plot(history.get('loss', []), label='Training Loss')
#     ax.plot(history.get('val_loss', []), label='Validation Loss')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.legend()
#     st.pyplot(fig)



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
import tempfile

# --- Paths to tokenizer and history ---
TOKENIZER_PATH = "tokenizer.pkl"
HISTORY_PATH = "train_history.pkl"
MAX_LENGTH = 34

# --- Streamlit page ---
st.set_page_config(page_title="📸 Image Caption Generator", layout="centered")
st.title("📸 Image Caption Generator (TensorFlow + Flickr8k)")
st.markdown("Upload your trained model (.h5) and an image to generate AI-powered captions!")

# --- Upload model ---
uploaded_model = st.file_uploader("Upload your trained model (.h5)", type=["h5"])
caption_model = None

if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        tmp_path = tmp.name
    try:
        caption_model = load_model(tmp_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

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

# --- Load tokenizer and feature extractor ---
tokenizer = load_tokenizer()
feature_model = load_feature_extractor()
history = load_history()

# --- File uploader for images ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and caption_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        tmp_img.write(uploaded_file.read())
        img_path = tmp_img.name

    image = Image.open(img_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🪄 Generate Caption"):
        with st.spinner("Generating caption..."):
            feature = extract_features(img_path, feature_model)
            caption = generate_caption(caption_model, tokenizer, feature, MAX_LENGTH)
            st.success("✨ Caption Generated!")
            st.markdown(f"**Generated Caption:** _{caption}_")
    
    os.remove(img_path)

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
