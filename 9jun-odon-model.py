import streamlit as st
import librosa
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import librosa.display
import requests

# Constants
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1iRMYXoIQfjDERuEKKD_IRDF-Q6a65nxR'
MODEL_PATH = 'odon-mos.pkl'
SAMPLE_RATE = 6000
DURATION = 1  # seconds

# Label mapping from folder to readable names
LABEL_ALIASES = {
    "d_17_02_08_21_36_05": "ae aegupti",
    "d_17_02_10_09_02_23": "ae. albopictus",
    "d_17_02_14_11_06_42": "an. arabiensis",
    "d_17_02_10_09_04_42": "an. gambiae",
    "d_17_02_14_11_10_29": "c. pipiens",
    "d_17_02_14_11_12_55": "c. quinquefasciatus"
}

# Mosquito info dictionary
MOSQUITO_INFO = {
    "ae aegupti": {
        "name": "🦟 Aedes aegypti",
        "description": "Spreads dengue, Zika, chikungunya, and yellow fever."
    },
    "ae. albopictus": {
        "name": "🦟 Aedes albopictus",
        "description": "Also known as the Asian tiger mosquito, transmits Zika, dengue, and chikungunya."
    },
    "an. arabiensis": {
        "name": "🦟 Anopheles arabiensis",
        "description": "A major malaria vector in sub-Saharan Africa."
    },
    "an. gambiae": {
        "name": "🦟 Anopheles gambiae",
        "description": "Primary vector of malaria in Africa."
    },
    "c. pipiens": {
        "name": "🦟 Culex pipiens",
        "description": "Transmits West Nile virus and filarial worms."
    },
    "c. quinquefasciatus": {
        "name": "🦟 Culex quinquefasciatus",
        "description": "Spreads West Nile virus and lymphatic filariasis."
    }
}

# Function to download model from Google Drive
def download_model(url, destination):
    if os.path.exists(destination):
        return  # Already downloaded
    with st.spinner("Downloading model... This may take a few minutes!"):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                st.progress(min(downloaded / total, 1.0))
    st.success("Model downloaded!")

# Feature extraction
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Plot waveform and spectrogram
def display_waveform_and_spectrogram(audio, sr):
    st.markdown("### 🎧 Audio Analysis")
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

# App setup
st.set_page_config(page_title="Mosquito Identifier", page_icon="🦟", layout="centered")
st.title("🦟 Mosquito Species Identifier")

st.markdown("""
Upload a **mosquito sound (.wav)** and this app will:
- Predict the **mosquito species**
- Display the **waveform & spectrogram**
- Show disease info for the **predicted species only**
""")

# Download model if not present
download_model(MODEL_URL, MODEL_PATH)

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📁 Upload mosquito sound (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, duration=DURATION)

    display_waveform_and_spectrogram(audio, sr)

    features = extract_features(audio, sr).reshape(1, -1)
    prediction = model.predict(features)[0]

    # Normalize prediction key
    formatted_key = prediction.strip().lower().replace('_', ' ')
    matched_code = None
    for alias_code in LABEL_ALIASES:
        if alias_code in formatted_key:
            matched_code = alias_code
            break

    if matched_code:
        species_key = LABEL_ALIASES.get(matched_code)
    else:
        species_key = LABEL_ALIASES.get(formatted_key, formatted_key)

    species_info_key = species_key.strip().lower()

    st.markdown(f"## ✅ **Predicted Species:** `{species_key.title()}`")

    info = MOSQUITO_INFO.get(species_info_key)
    if info:
        st.markdown("### 🧬 Mosquito Species & Disease Info")
        st.markdown("---")
        st.markdown(f"#### 🟢 **{info['name']}** *(Predicted)*")
        st.markdown(f"- **Disease Info**: {info['description']}")
    else:
        st.info("No information found for the predicted species.")

footer = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            text-align: center;
            padding: 15px;
            font-size: 14px;
            font-weight: bold;
            color: #333;
            border-top: 1px solid #ddd;
            z-index: 9999;
        }
    </style>
    <div class="footer">
        📢 <b>This analysis is for educational purposes only.</b><br>
        🏫 <b>Supervised by:</b> Dr. Valerie Odon, University of Strathclyde, UK<br>
        💻 <b>Developed by:</b> Odon’s Lab, PhD Students<br>
        📌 <i>Note: All data and resources used are publicly available.</i>
    </div>
"""

st.markdown("<div style='padding-bottom: 100px;'></div>", unsafe_allow_html=True)
st.markdown(footer, unsafe_allow_html=True)
