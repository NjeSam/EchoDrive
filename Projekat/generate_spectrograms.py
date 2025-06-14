
import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
from scipy.ndimage import zoom

# Podešavanja
WAV_DIR = 'wav_commands'
SPECTROGRAM_DIR = 'spectrograms'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3  # sekunde
TARGET_LEN = SAMPLE_RATE * DURATION

# Mape komandi
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

def process_and_save_spectrograms():
    expected_shape = (N_MELS, int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1)

    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Generating spectrograms"):
            if not filename.endswith('.wav'): continue
            cmd = extract_command(filename)
            if cmd is None: continue

            wav_path = os.path.join(root, filename)
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

            # Trimuj/paduj na tačno 3 sekunde
            if len(y) > TARGET_LEN:
                y = y[:TARGET_LEN]
            elif len(y) < TARGET_LEN:
                y = np.pad(y, (0, TARGET_LEN - len(y)))

            spec = create_spectrogram(y, sr)
            if spec.shape != expected_shape:
                zoom_factors = (expected_shape[0] / spec.shape[0], expected_shape[1] / spec.shape[1])
                spec = zoom(spec, zoom_factors)

            out_path = os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_'), filename.replace('.wav', '.npy'))
            np.save(out_path, spec)

if __name__ == "__main__":
    process_and_save_spectrograms()
