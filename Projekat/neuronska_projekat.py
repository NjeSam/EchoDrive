# Kompletna verzija sa augmentacijom i fiksiranjem duzine zvuka
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import zoom
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth


WAV_DIR = 'wav_commands'
SPECTROGRAM_DIR = 'spectrograms'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3  # sekunde

command_mapping = {
    'AI ON': 0, 'AI OFF': 1, 'TURN ON THE AIR CONDITIONER': 2, 'TURN OFF THE AIR CONDITIONER': 3,
    'TURN ON THE RADIO': 4, 'TURN OFF THE RADIO': 5, 'SWITCH THE RADIO STATION': 6,
    'MUTE THE RADIO': 7, 'TURN UP THE VOLUME': 8, 'TURN DOWN THE VOLUME': 9,
    'TURN ON THE NAVIGATION': 10, 'TURN OFF THE NAVIGATION': 11
}

os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
for cmd in command_mapping:
    os.makedirs(os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_')), exist_ok=True)

def extract_command(filename):
    filename = filename.replace('.wav', '').upper()
    for cmd in sorted(command_mapping.keys(), key=lambda x: -len(x)):
        if filename.startswith(cmd):
            return cmd
    return None

def augment_audio(y, sr):
    aug = []
    aug.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-1.5, 1.5)))
    aug.append(librosa.effects.time_stretch(y, rate=random.uniform(0.85, 1.15)))
    noise = np.random.normal(0, 0.005, y.shape)
    aug.append(y + noise)
    return aug

def create_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())
    return norm_S

def process_all_files():
    features, labels = [], []
    expected_shape = (N_MELS, int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1)
    target_len = SAMPLE_RATE * DURATION

    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Processing files"):
            if not filename.endswith('.wav'): continue
            cmd = extract_command(filename)
            if cmd is None: continue

            wav_path = os.path.join(root, filename)
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

            all_versions = [y] + augment_audio(y, sr)
            for y_aug in all_versions:
                # Trimuj/paduj na tačno 3 sekunde
                if len(y_aug) > target_len:
                    y_aug = y_aug[:target_len]
                elif len(y_aug) < target_len:
                    y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))

                spec = create_spectrogram(y_aug, sr)
                if spec.shape != expected_shape:
                    zoom_factors = (expected_shape[0] / spec.shape[0], expected_shape[1] / spec.shape[1])
                    spec = zoom(spec, zoom_factors)
                features.append(spec)
                labels.append(command_mapping[cmd])

    features = np.array(features)[..., np.newaxis]
    labels = np.array(labels)
    return features, labels

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig('training_metrics.png'); plt.show()


def play_music():
    sp.start_playback()
    print("▶️ Glasba se predvaja.")

def pause_music():
    sp.pause_playback()
    print("⏸️ Glasba je na pavzi.")

def next_track():
    sp.next_track()
    print("⏭️ Naslednja pesem.")


def volume_up():
    current = sp.current_playback()
    if current and 'device' in current:
        current_volume = current['device']['volume_percent']
        sp.volume(min(100, current_volume + 10))
        print(f"🔊 Glasnost zvišana na {min(100, current_volume + 10)}%.")

def volume_down():
    current = sp.current_playback()
    if current and 'device' in current:
        current_volume = current['device']['volume_percent']
        sp.volume(max(0, current_volume - 10))
        print(f"🔉 Glasnost zmanjšana na {max(0, current_volume - 10)}%.")


if __name__ == "__main__":

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="d5b6a52f2b1b47288972ddf296b46ccd",
    client_secret="b68ef098e4b1457d9fac6d5a79ec4ce7",
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state"
    ))


    print("Opcije:\n1 - Trenutni trening\n2 - Samo predikcija (untitled.wav)")
    izbor = input("Unesi 1 ili 2: ").strip()
    if izbor == "1":
        X, y = process_all_files()
        y_cat = to_categorical(y, num_classes=len(command_mapping))
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

        model = build_model(X.shape[1:], len(command_mapping))
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
        history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                            callbacks=[early_stop, reduce_lr])
        model.save("keras_model.h5")
        plot_training_history(history)
        print("\n✅ Model sačuvan kao 'keras_model.h5'.")

    elif izbor == "2":
        if not os.path.exists("AI OFF.wav"):
            print("❌ Nema fajla 'untitled.wav'.")
        else:
            y, sr = librosa.load("TURN OFF THE RADIO LAN D 4.wav", sr=SAMPLE_RATE, duration=DURATION)
            if len(y) < SAMPLE_RATE * DURATION:
                y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
            spec = create_spectrogram(y, sr)
            expected_shape = (N_MELS, int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1)
            if spec.shape != expected_shape:
                zoom_factors = (expected_shape[0] / spec.shape[0], expected_shape[1] / spec.shape[1])
                spec = zoom(spec, zoom_factors)
            x_input = spec[np.newaxis, ..., np.newaxis]
            model = load_model("keras_model.h5")
            pred = np.argmax(model.predict(x_input))
            inv_map = {v: k for k, v in command_mapping.items()}
            recognized_cmd = inv_map[pred]
            print(f"🎧 Prepoznata komanda: {recognized_cmd}")
            

            if recognized_cmd == "TURN ON THE RADIO":
                play_music()
            if recognized_cmd == "TURN OFF THE RADIO":
                pause_music()
            if recognized_cmd == "SWITCH THE RADIO STATION":
                next_track()
            if recognized_cmd == "TURN DOWN THE VOLUME":
                volume_down()
            if recognized_cmd == "TURN UP THE VOLUME":
                volume_up()