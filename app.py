import os
import io
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add
from tensorflow.keras.utils import get_custom_objects

# ----------------------
# CONFIG
# ----------------------

MODEL_WEIGHTS_PATH = "best_model2.h5"  # your H5 weights file
TOKENIZER_PATH = "tokenizer.pkl"
FEATURES_PATH = "features.pkl"
DEFAULT_MAX_LENGTH = 34  # fallback if tokenizer max length is not set
EMBED_DIM = 256  # adjust to your model
LSTM_UNITS = 256  # adjust to your model

# ----------------------
# HELPERS
# ----------------------

@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    """Return VGG16 up to fc2 (4096-d) for on-the-fly feature extraction."""
    base = VGG16()
    feat_extractor = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    return feat_extractor

@st.cache_data(show_spinner=False)
def load_tokenizer(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

@st.cache_data(show_spinner=False)
def load_features(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_caption_model(vocab_size, max_length):
    """Rebuild the image captioning model architecture."""
    # Image feature input
    inputs1 = Input(shape=(4096,))
    fe1 = Dense(EMBED_DIM, activation='relu')(inputs1)

    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs2)
    se2 = LSTM(LSTM_UNITS)(se1)

    # Merge
    decoder1 = Add()([fe1, se2])
    decoder2 = Dense(EMBED_DIM, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def extract_features_from_pil(pil_img, feat_extractor):
    img = pil_img.resize((224,224))
    arr = img_to_array(img)
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    arr = preprocess_input(arr)
    feat = feat_extractor.predict(arr, verbose=0)
    return feat

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# ----------------------
# STREAMLIT APP UI
# ----------------------

st.set_page_config(page_title='Image Captioning (Flickr8k)', layout='centered')
st.title('Image Captioning — Flickr8k model')
st.markdown('Upload an image and the app will generate a caption using the trained model.')

# Load tokenizer and features
tokenizer = load_tokenizer(TOKENIZER_PATH)
if tokenizer is None:
    st.error("Tokenizer not found. Please place tokenizer.pkl in the app folder.")
    st.stop()
vocab_size = len(tokenizer.word_index) + 1
max_length = getattr(tokenizer, 'max_length', DEFAULT_MAX_LENGTH)

features_db = load_features(FEATURES_PATH)
if features_db:
    st.success('Precomputed features loaded: %d images' % len(features_db))

# Feature extractor
feat_extractor = load_feature_extractor()

# Build model architecture and load weights
model = build_caption_model(vocab_size, max_length)
model.load_weights(MODEL_WEIGHTS_PATH)
st.success("Captioning model loaded from architecture + weights.")

# Upload image
uploaded = st.file_uploader('Upload an image (jpg/png)', type=['jpg','jpeg','png'])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
        st.image(image, caption='Uploaded image', use_column_width=True)
    except Exception as e:
        st.error(f'Could not read image: {e}')
        image = None

    if image is not None:
        with st.spinner('Extracting features and predicting caption...'):
            feat = extract_features_from_pil(image, feat_extractor)
            image_feat = feat.reshape((1, -1)) if feat.ndim != 2 else feat
            caption = predict_caption(model, image_feat, tokenizer, max_length)
        st.subheader('Predicted caption')
        st.write(caption)
        st.success('Done')
else:
    st.info('Upload an image to generate a caption.')

# Footer / troubleshooting
st.markdown('---')
# st.write('Tips:')
# st.write('- Place `best_model.h5` and `tokenizer.pkl` in the working directory.')
# st.write('- If you trained using precomputed features, you can provide `features.pkl` or let the app extract features.')
st.write('- Loading may take a few seconds due to VGG16 initialization.')
