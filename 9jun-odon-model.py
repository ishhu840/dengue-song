import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import os # Added for os.remove

# ✅ Constants
MODEL_PATH = str(Path.home() / "Documents" / "odon-mos-cnn.pth")
LABEL_PATH = str(Path.home() / "Documents" / "odon-mos-labelsnew.pkl")
SAMPLE_RATE = 16000
CLIP_DURATION = 1

# ✅ Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ✅ Define the SAME model as used in training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Label mapping from folder to readable names (Copied from 9jun-odon-model.py)
LABEL_ALIASES = {
    "d_17_02_08_21_36_05": "ae aegypti",
    "d_17_02_10_09_02_23": "ae. albopictus",
    "d_17_02_14_11_06_42": "an. arabiensis",
    "d_17_02_10_09_04_42": "an. gambiae",
    "d_17_02_14_11_10_29": "c. pipiens",
    "d_17_02_14_11_12_55": "c. quinquefasciatus"
}

# Mosquito info (Copied from 9jun-odon-model.py)
MOSQUITO_INFO = {
    "ae aegypti": {
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


# ✅ Load model and label encoder
@st.cache_resource
def load_model_and_labels():
    label_encoder = joblib.load(LABEL_PATH)
    model = SimpleCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, label_encoder


def preprocess_audio(audio_bytes):
    # librosa.load expects a file path or file-like object, not raw bytes directly
    # For Streamlit uploaded_file, it's a BytesIO object, which librosa can handle.
    y, _ = librosa.load(audio_bytes, sr=SAMPLE_RATE, duration=CLIP_DURATION)
    if len(y) < SAMPLE_RATE * CLIP_DURATION:
        pad = SAMPLE_RATE * CLIP_DURATION - len(y)
        y = np.pad(y, (0, pad))
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) # Added epsilon to avoid division by zero
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
    mel_tensor = F.interpolate(mel_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0) # Added align_corners=False
    return mel_tensor


def plot_waveform_and_spectrogram(y, sr):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=axs[0]) # Changed axs[0].plot(y) to librosa.display.waveshow
    axs[0].set_title("Waveform")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[1], cmap='viridis') # Used librosa.display.specshow
    axs[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB') # Added format for colorbar
    st.pyplot(fig)


# ✅ Streamlit UI
st.set_page_config(page_title="Mosquito Identifier", page_icon="🦟", layout="centered") # Added page config
st.title("🦟 Mosquito Species Identifier")
st.markdown("""
Upload a **mosquito sound (.wav)** and this app will:
- Predict the **mosquito species**
- Display the **waveform & spectrogram**
- Show disease info for the **predicted species only**
""")

model, label_encoder = load_model_and_labels()

uploaded_file = st.file_uploader("📁 Upload mosquito sound (.wav)", type=["wav"])

if uploaded_file:
    # Save uploaded file to a temporary location for librosa to load
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    y, sr = librosa.load("temp_audio.wav", sr=SAMPLE_RATE) # Load from temp file
    st.audio(uploaded_file, format='audio/wav')

    plot_waveform_and_spectrogram(y, sr)

    # Preprocess
    input_tensor = preprocess_audio(uploaded_file).unsqueeze(0).to(DEVICE) # Pass uploaded_file directly

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        predicted_raw_label = label_encoder.inverse_transform([predicted_index])[0]

    # Normalize prediction using the LABEL_ALIASES structure
    formatted_key = predicted_raw_label.strip().lower().replace('_', ' ')
    predicted_species = LABEL_ALIASES.get(formatted_key, formatted_key)

    st.write(f"Predicted Mosquito Species: **{predicted_species.title()}**")

    # Display disease information (integrated directly)
    species_info_key = predicted_species.strip().lower()
    info = MOSQUITO_INFO.get(species_info_key)
    if info:
        st.markdown("### 🧬 Mosquito Species & Disease Info")
        st.markdown("---")
        st.markdown(f"#### 🟢 **{info['name']}** *(Predicted)*")
        st.markdown(f"- **Disease Info**: {info['description']}")
    else:
        st.info("No information found for the predicted species.")

    # Clean up temporary file
    os.remove("temp_audio.wav")

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("<div style='padding-bottom: 100px;'></div>", unsafe_allow_html=True)
