import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Konfiguracija
WAV_DIR = 'wav_commands'
SPECTROGRAM_DIR = 'spectrograms'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 2  # u sekundama

# Mapa komandi
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

# Kreiraj izlazne foldere
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
        y = np.pad(y, (0, max(0, SAMPLE_RATE * DURATION - len(y))), mode='constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())

    if save_path:
        np.save(save_path, norm_S)
        if not os.path.exists(save_path):
            print(f"❌ Greška pri snimanju: {save_path}")

    if show:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spektrogram: {os.path.basename(audio_path)}')
        plt.tight_layout()
        plt.show()

    return norm_S

def process_all_files():
    features, labels = [], []
    skipped_files = []

    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Obrada fajlova"):
            if filename.endswith('.wav'):
                cmd = extract_command(filename)
                if cmd is None:
                    print(f"Preskočen fajl: {filename}")
                    skipped_files.append(filename)
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

def load_spectrograms():
    features, labels = [], []
    for cmd, label in command_mapping.items():
        cmd_dir = os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_'))
        if not os.path.exists(cmd_dir):
            continue
        for file in os.listdir(cmd_dir):
            if file.endswith('.npy'):
                path = os.path.join(cmd_dir, file)
                spectrogram = np.load(path)
                features.append(spectrogram)
                labels.append(label)
    return np.array(features), np.array(labels)

def spectrogram_files_exist():
    for cmd in command_mapping:
        cmd_folder = os.path.join(SPECTROGRAM_DIR, cmd.replace(' ', '_'))
        if os.path.exists(cmd_folder):
            if any(f.endswith('.npy') for f in os.listdir(cmd_folder)):
                return True
    return False

# MLP implementacija u NumPy-u
class SimpleCNN:
    def __init__(self, input_shape, num_classes, lr=0.01):
        self.lr = lr
        self.num_classes = num_classes
        self.input_shape = input_shape  # (1, H, W)

        # Konvolucioni sloj 1
        self.k1 = np.random.randn(8, 1, 3, 3) * np.sqrt(2 / 9)
        self.b1 = np.zeros((8, 1))

        # Konvolucioni sloj 2
        self.k2 = np.random.randn(16, 8, 3, 3) * np.sqrt(2 / (8 * 9))
        self.b2 = np.zeros((16, 1))

        # Dimenzije posle konvolucije i pooling-a (računa se kasnije)
        self.fc_input_size = None

        # Potpuno povezani sloj (inicijalizuje se nakon prve propagacije)
        self.fc_w = None
        self.fc_b = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def conv2d(self, x, kernel, bias):
        C_out, C_in, kh, kw = kernel.shape
        _, h, w = x.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        out = np.zeros((C_out, out_h, out_w))
        for co in range(C_out):
            for ci in range(C_in):
                for i in range(out_h):
                    for j in range(out_w):
                        out[co, i, j] += np.sum(x[ci, i:i+kh, j:j+kw] * kernel[co, ci])
            out[co] += bias[co]
        return out

    def max_pool(self, x, size=2):
        C, h, w = x.shape
        out_h = h // size
        out_w = w // size
        out = np.zeros((C, out_h, out_w))
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    out[c, i, j] = np.max(x[c, i*size:(i+1)*size, j*size:(j+1)*size])
        return out

    def forward(self, x):
        self.conv1 = self.conv2d(x, self.k1, self.b1)
        self.act1 = self.relu(self.conv1)
        self.pool1 = self.max_pool(self.act1)

        self.conv2 = self.conv2d(self.pool1, self.k2, self.b2)
        self.act2 = self.relu(self.conv2)
        self.pool2 = self.max_pool(self.act2)

        self.flat = self.pool2.reshape(-1)

        # Inicijalizuj FC sloj tek sada ako nije
        if self.fc_w is None:
            self.fc_input_size = self.flat.shape[0]
            self.fc_w = np.random.randn(self.fc_input_size, self.num_classes) * np.sqrt(2 / self.fc_input_size)
            self.fc_b = np.zeros((1, self.num_classes))

        self.logits = self.flat @ self.fc_w + self.fc_b
        self.probs = self.softmax(self.logits.reshape(1, -1))
        return self.probs

    def backward(self, x, y_true):
        y_true_onehot = np.zeros_like(self.probs)
        y_true_onehot[0, y_true] = 1
        dlogits = self.probs - y_true_onehot  # shape: (1, num_classes)

        dW = self.flat.reshape(-1, 1) @ dlogits
        db = dlogits

        self.fc_w -= self.lr * dW
        self.fc_b -= self.lr * db

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            for i in range(len(X)):
                probs = self.forward(X[i])
                pred = np.argmax(probs)
                loss = -np.log(probs[0, y[i]] + 1e-8)
                total_loss += loss
                correct += int(pred == y[i])
                self.backward(X[i], y[i])
            acc = correct / len(X)
            print(f"Epoch {epoch+1}: Loss={total_loss/len(X):.4f}, Accuracy={acc:.4f}")

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)


def predict_wav_with_cnn(wav_file):
    spectrogram = create_spectrogram(wav_file)
    x_input = spectrogram[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    cnn = SimpleCNN(input_shape=(1, spectrogram.shape[0], spectrogram.shape[1]), num_classes=len(command_mapping))
    cnn.load('cnn_weights.npz')
    pred_idx = cnn.predict(x_input[0])
    reverse_mapping = {v: k for k, v in command_mapping.items()}
    print(f"\n🎧 Predikcija: {reverse_mapping[pred_idx]}")
def save(self, path='cnn_weights.npz'):
    np.savez(path, fc_w=self.fc_w, fc_b=self.fc_b)

def load(self, path='cnn_weights.npz'):
    data = np.load(path)
    self.fc_w = data['fc_w']
    self.fc_b = data['fc_b']



if __name__ == "__main__":
    print("\n🎤 Glasovna komanda – MLP klasifikator")
    print("1. Trenira model")
    print("2. Testira fajl 'untitled.wav'")
    choice = input("\n👉 Unesi izbor (1 ili 2): ").strip()

    if choice == "1":
        print("\n🚀 Početak obrade podataka...")

        if spectrogram_files_exist():
            print("\n📥 Učitavam postojeće spektrograme...")
            X, y = load_spectrograms()
        else:
            print("\n🔄 Kreiram nove spektrograme...")
            X, y = process_all_files()

        if len(X) == 0:
            print("\n❌ KRITIČNA GREŠKA: Nema obrađenih podataka!")
        else:
            X_reshaped = X[:, np.newaxis, :, :]  # Dodaj kanal: (N, 1, H, W)
            cnn = SimpleCNN(input_shape=(1, X.shape[1], X.shape[2]), num_classes=len(command_mapping), lr=0.01)
            cnn.train(X_reshaped, y, epochs=10)


            print("\n💾 Model sačuvan u 'mlp_weights.npz'")

    elif choice == "2":
        test_file = "untitled.wav"
        if os.path.exists(test_file):
            print(f"\n🔍 Testiram fajl: {test_file}")
            predict_wav_with_cnn(test_file)
        else:
            print(f"\n❗ Test fajl '{test_file}' nije pronađen u trenutnom folderu.")
    else:
        print("\n❗ Nevažeći unos. Pokreni program ponovo i unesi 1 ili 2.")
