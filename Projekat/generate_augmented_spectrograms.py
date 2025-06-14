
import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
from scipy.ndimage import zoom
import random

# Podešavanja
WAV_DIR = 'wav_commands'
SPECTROGRAM_DIR = 'spectrograms_augmented'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3  # sekunde
TARGET_LEN = SAMPLE_RATE * DURATION

# Mapa komandi
command_mapping = {
    'AI ON': 0, 'AI OFF': 1, 'TURN ON THE AIR CONDITIONER': 2, 'TURN OFF THE AIR CONDITIONER': 3,
    'TURN ON THE RADIO': 4, 'TURN OFF THE RADIO': 5, 'SWITCH THE RADIO STATION': 6,
    'MUTE THE RADIO': 7, 'TURN UP THE VOLUME': 8, 'TURN DOWN THE VOLUME': 9,
    'TURN ON THE NAVIGATION': 10, 'TURN OFF THE NAVIGATION': 11
}

# Priprema foldera
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
for cmd in command_mapping:
    os.makedirs(os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_')), exist_ok=True)

def extract_command(filename):
    filename = filename.replace('.wav', '').upper()
    for cmd in sorted(command_mapping.keys(), key=lambda x: -len(x)):
        if filename.startswith(cmd):
            return cmd
    return None

def create_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())
    return norm_S

def augment_audio(y, sr):
    aug = []
    aug.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-1.5, 1.5)))
    aug.append(librosa.effects.time_stretch(y, rate=random.uniform(0.85, 1.15)))
    noise = np.random.normal(0, 0.005, y.shape)
    aug.append(y + noise)
    return aug

def process_and_save_augmented_spectrograms():
    expected_shape = (N_MELS, int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1)

    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Generating augmented spectrograms"):
            if not filename.endswith('.wav'): continue
            cmd = extract_command(filename)
            if cmd is None: continue

            wav_path = os.path.join(root, filename)
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

            versions = [y] + augment_audio(y, sr)
            for i, y_aug in enumerate(versions):
                # Trimuj/paduj na tačno 3 sekunde
                if len(y_aug) > TARGET_LEN:
                    y_aug = y_aug[:TARGET_LEN]
                elif len(y_aug) < TARGET_LEN:
                    y_aug = np.pad(y_aug, (0, TARGET_LEN - len(y_aug)))

                spec = create_spectrogram(y_aug, sr)
                if spec.shape != expected_shape:
                    zoom_factors = (expected_shape[0] / spec.shape[0], expected_shape[1] / spec.shape[1])
                    spec = zoom(spec, zoom_factors)

                suffix = f'_aug{i}' if i > 0 else ''
                out_path = os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_'), filename.replace('.wav', f'{suffix}.npy'))
                np.save(out_path, spec)

if __name__ == "__main__":
    process_and_save_augmented_spectrograms()
