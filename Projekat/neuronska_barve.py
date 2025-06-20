import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parametri
WAV_DIR = "C:/Users/ACER/Documents/GitHub/EchoDrive/Projekat/wav_commands"
MODEL_PATH = "govornik_model.h5"
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
HOP_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.6
MARGIN_THRESHOLD = 0.2
LABELS = ['DJORDJE', 'LAN', 'NJEGOS']
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

# Funkcija za generisanje spektrograma
def create_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < DURATION * sr:
        y = np.pad(y, (0, DURATION * sr - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db / 80.0 + 1  # normalizacija u [0,1]
    return mel_db.astype(np.float32)

# Treniranje modela
def train_model():
    X, y = [], []
    for fname in os.listdir(WAV_DIR):
        if not fname.endswith(".wav"):
            continue
        label = next((l for l in LABELS if l in fname.upper()), None)
        if label:
            spec = create_spectrogram(os.path.join(WAV_DIR, fname))
            X.append(spec)
            y.append(LABEL_TO_INDEX[label])

    if not X:
        print("\n❌ Nema dovoljno podataka za treniranje.")
        return

    X = np.array(X)[..., np.newaxis]
    y = to_categorical(y, num_classes=len(LABELS))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(N_MELS, X.shape[2], 1)),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(LABELS), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_val, y_val))

    model.save(MODEL_PATH)
    print("\n✅ Model je uspešno istreniran i sačuvan.\n")

# Predikcija govornika sa dodatnom proverom sigurnosti
def predict_speaker(file_path):
    if not os.path.exists(MODEL_PATH):
        print("❌ Nema modela. Pokreni opciju 1 za treniranje.")
        return

    model = load_model(MODEL_PATH)
    spec = create_spectrogram(file_path)
    spec = np.expand_dims(spec, axis=(0, -1))
    preds = model.predict(spec)[0]

    sorted_indices = np.argsort(preds)[::-1]
    conf1 = preds[sorted_indices[0]]
    conf2 = preds[sorted_indices[1]]
    label_idx = sorted_indices[0]

    # Odluka na osnovu poverenja i margine
    if conf1 < CONFIDENCE_THRESHOLD or (conf1 - conf2) < MARGIN_THRESHOLD:
        label = "NEPOZNATO"
    else:
        label = LABELS[label_idx]

    print(f"\n🎤 Govornik: {label}")
    print(f"📊 Pouzdanost: {conf1:.2f}, Margin: {conf1 - conf2:.2f}\n")

# Glavni meni
if __name__ == "__main__":
    print("== GOVORNIK PREPOZNAVANJE ==")
    print("1 - Treniraj model")
    print("2 - Testiraj fajl 'Testni.wav'\n")

    izbor = input("Izaberi (1/2): ").strip()

    if izbor == "1":
        train_model()
    elif izbor == "2":
        test_path = os.path.join(WAV_DIR, "AI OFF LUKA.wav")
        if os.path.exists(test_path):
            predict_speaker(test_path)
        else:
            print("\n❌ Fajl Testni.wav nije pronađen u folderu.\n")
    else:
        print("\n❌ Nepoznata opcija.\n")
