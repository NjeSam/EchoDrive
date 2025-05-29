import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Configuration
WAV_DIR = 'wav_commands'
SPECTROGRAM_DIR = 'spectrograms'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3  # seconds

# Command mapping
command_mapping = {
    'AI ON': 0,
    'AI OFF': 1,
    'TURN ON THE AIR CONDITIONER': 2,
    'TURN OFF THE AIR CONDITIONER': 3,
    'TURN ON THE RADIO': 4,
    'TURN OFF THE RADIO': 5,
    'SWITCH THE RADIO STATION': 6,
    'MUTE THE RADIO': 7,
    'TURN UP THE VOLUME': 8,
    'TURN DOWN THE VOLUME': 9,
    'TURN ON THE NAVIGATION': 10,
    'TURN OFF THE NAVIGATION': 11
}

# Custom class weights
class_weight = {
    0: 1.0,  # AI ON
    1: 1.0,  # AI OFF
    2: 1.0,  # TURN ON THE AIR CONDITIONER
    3: 1.0,  # TURN OFF THE AIR CONDITIONER
    4: 1.0,  # TURN ON THE RADIO
    5: 1.0,  # TURN OFF THE RADIO
    6: 1.0,  # SWITCH THE RADIO STATION
    7: 1.0,  # MUTE THE RADIO
    8: 1.0,  # TURN UP THE VOLUME
    9: 1.0,  # TURN DOWN THE VOLUME
    10: 1.0, # TURN ON THE NAVIGATION
    11: 1.0  # TURN OFF THE NAVIGATION
}

# Create output directories
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
for cmd in command_mapping:
    os.makedirs(os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_')), exist_ok=True)

def extract_command(filename):
    filename = filename.replace('.wav', '').upper()
    for cmd in sorted(command_mapping.keys(), key=lambda x: -len(x)):
        if filename.startswith(cmd):
            return cmd
    return None

def create_spectrogram(audio_path, save_path=None, show=False):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())

    if save_path:
        np.save(save_path, norm_S)

    if show:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {os.path.basename(audio_path)}')
        plt.tight_layout()
        plt.show()

    return norm_S

def process_all_files():
    features, labels = [], []
    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Processing files"):
            if filename.endswith('.wav'):
                cmd = extract_command(filename)
                if cmd is None:
                    continue
                wav_path = os.path.join(root, filename)
                cmd_folder = cmd.replace(' ', '_')
                spec_filename = filename.replace('.wav', '.npy')
                spec_path = os.path.join(SPECTROGRAM_DIR, cmd_folder, spec_filename)

                if os.path.exists(spec_path):
                    spectrogram = np.load(spec_path)
                else:
                    spectrogram = create_spectrogram(wav_path, save_path=spec_path)

                features.append(spectrogram)
                labels.append(command_mapping[cmd])

    return np.array(features), np.array(labels)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_wav_with_model(wav_file, model_path='keras_model.h5'):
    spectrogram = create_spectrogram(wav_file)

    expected_height = 128
    expected_width = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1
    if spectrogram.shape != (expected_height, expected_width):
        from scipy.ndimage import zoom
        zoom_factors = (expected_height/spectrogram.shape[0], expected_width/spectrogram.shape[1])
        spectrogram = zoom(spectrogram, zoom_factors)

    x_input = spectrogram[np.newaxis, ..., np.newaxis]
    model = load_model(model_path)
    pred_idx = np.argmax(model.predict(x_input))
    reverse_mapping = {v: k for k, v in command_mapping.items()}
    print(f"\n\U0001F3A7 Prediction: {reverse_mapping[pred_idx]}")

if __name__ == "__main__":
    print("\n\U0001F3A4 Voice Command Classifier (Keras)")
    print("1. Train model")
    print("2. Test 'untitled.wav'")
    choice = input("\n\U0001F449 Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\n\U0001F680 Starting model training...")
        X, y = process_all_files()

        expected_height = N_MELS
        expected_width = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

        from scipy.ndimage import zoom
        X_resized = []
        for spec in X:
            if spec.shape != (expected_height, expected_width):
                zoom_factors = (expected_height/spec.shape[0], expected_width/spec.shape[1])
                spec = zoom(spec, zoom_factors)
            X_resized.append(spec[..., np.newaxis])
        X_resized = np.array(X_resized)

        y_indices = y  # bez one-hot enkodovanja

        X_train, X_test, y_train, y_test = train_test_split(X_resized, y_indices, test_size=0.2, random_state=42, stratify=y_indices)

        model = build_model(input_shape=(expected_height, expected_width, 1), num_classes=len(command_mapping))
        model.fit(X_train, to_categorical(y_train),
                  epochs=100,
                  batch_size=8,
                  validation_data=(X_test, to_categorical(y_test)),
                  class_weight=class_weight)

        model.save("keras_model.h5")
        print("\n📀 Model saved to 'keras_model.h5'")

    elif choice == "2":
        test_file = "untitled.wav"
        if os.path.exists(test_file):
            predict_wav_with_model(test_file)
        else:
            print(f"[ERROR] File '{test_file}' not found.")
    else:
        print("[ERROR] Invalid choice. Enter 1 or 2.") 