import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        y = np.pad(y, (0, max(0, SAMPLE_RATE * DURATION - len(y))), mode='constant')

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
    skipped_files = []

    for root, _, files in os.walk(WAV_DIR):
        for filename in tqdm(files, desc="Processing files"):
            if filename.endswith('.wav'):
                cmd = extract_command(filename)
                if cmd is None:
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

class SimpleCNN:
    def __init__(self, input_shape, num_classes, lr=0.01):
        self.lr = lr
        self.lr_decay = 0.95
        self.num_classes = num_classes
        self.input_shape = input_shape  # Should be (1, height, width)
        self.dropout_rate = 0.2

        # Conv Layer 1
        self.k1 = np.random.randn(8, 1, 3, 3) * np.sqrt(2 / 9)
        self.b1 = np.zeros((8, 1))

        # Conv Layer 2
        self.k2 = np.random.randn(16, 8, 3, 3) * np.sqrt(2 / (8 * 9))
        self.b2 = np.zeros((16, 1))

        # Calculate the flattened size
        self._calculate_fc_input_size()

    def _calculate_fc_input_size(self):
        # Mock forward pass to calculate flattened size
        x = np.zeros((1, *self.input_shape))
        x = self.conv2d(x, self.k1, self.b1)
        x = self.max_pool(x)
        x = self.conv2d(x, self.k2, self.b2)
        x = self.max_pool(x)
        self.fc_input_size = x.reshape(-1).shape[0]
        
        # Initialize FC layer
        self.fc_w = np.random.randn(self.fc_input_size, self.num_classes) * np.sqrt(2 / self.fc_input_size)
        self.fc_b = np.zeros((1, self.num_classes))

    def relu(self, x):
        return np.maximum(0, x)

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

    def forward(self, x, training=True):
        self.conv1 = self.conv2d(x, self.k1, self.b1)
        self.act1 = self.relu(self.conv1)
        if training:
            self.act1 *= (1 - self.dropout_rate)  # Dropout
        self.pool1 = self.max_pool(self.act1)

        self.conv2 = self.conv2d(self.pool1, self.k2, self.b2)
        self.act2 = self.relu(self.conv2)
        if training:
            self.act2 *= (1 - self.dropout_rate)  # Dropout
        self.pool2 = self.max_pool(self.act2)

        self.flat = self.pool2.reshape(-1)

        if self.fc_w is None:
            self.fc_input_size = self.flat.shape[0]
            self.fc_w = np.random.randn(self.fc_input_size, self.num_classes) * np.sqrt(2 / self.fc_input_size)
            self.fc_b = np.zeros((1, self.num_classes))

        self.logits = self.flat @ self.fc_w + self.fc_b
        self.probs = self.softmax(self.logits.reshape(1, -1))
        return self.probs

    def backward(self, x, y_true):
        # Label smoothing
        y_true_onehot = np.full((1, self.num_classes), 0.1 / (self.num_classes - 1))
        y_true_onehot[0, y_true] = 0.9

        dlogits = self.probs - y_true_onehot

        dW = self.flat.reshape(-1, 1) @ dlogits
        db = dlogits

        # Gradient clipping
        dW = np.clip(dW, -1.0, 1.0)
        db = np.clip(db, -1.0, 1.0)

        self.fc_w -= self.lr * dW
        self.fc_b -= self.lr * db

    def train(self, X, y, epochs=10):
        print("[INFO] Starting CNN training")
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            print(f"\n[INFO] --- Epoch {epoch+1}/{epochs} ---")
            
            for i in range(len(X)):
                probs = self.forward(X[i])
                pred = np.argmax(probs)
                
                # Stable loss calculation
                probs_clipped = np.clip(probs, 1e-10, 1.0 - 1e-10)
                loss = -np.log(probs_clipped[0, y[i]])
                total_loss += loss
                
                correct += int(pred == y[i])
                self.backward(X[i], y[i])
                
                if i % 10 == 0:
                    print(f"[DEBUG] Sample {i}: Loss={loss:.4f}, Pred={pred}, True={y[i]}")
            
            # Learning rate decay
            self.lr *= self.lr_decay
            
            acc = correct / len(X)
            avg_loss = total_loss / len(X)
            print(f"[RESULT] Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
            print(f"[INFO] New learning rate: {self.lr:.6f}")

    def save(self, path='cnn_weights.npz'):
        np.savez(path,
                k1=self.k1, b1=self.b1,
                k2=self.k2, b2=self.b2,
                fc_w=self.fc_w, fc_b=self.fc_b)

    def load(self, path='cnn_weights.npz'):
        data = np.load(path)
        self.k1 = data['k1']
        self.b1 = data['b1']
        self.k2 = data['k2']
        self.b2 = data['b2']
        self.fc_w = data['fc_w']
        self.fc_b = data['fc_b']

    
    def predict(self, x):
        """Added predict method that wraps forward pass"""
        probs = self.forward(x, training=False)  # Set training=False for prediction
        return np.argmax(probs)

def predict_wav_with_cnn(wav_file):
    spectrogram = create_spectrogram(wav_file)
    
    # Ensure the spectrogram has expected dimensions
    expected_height = 128  # Should match N_MELS
    expected_width = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1
    
    if spectrogram.shape != (expected_height, expected_width):
        # Resize if necessary
        from scipy.ndimage import zoom
        zoom_factors = (expected_height/spectrogram.shape[0], 
                       expected_width/spectrogram.shape[1])
        spectrogram = zoom(spectrogram, zoom_factors)
    
    x_input = spectrogram[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
    
    cnn = SimpleCNN(input_shape=(expected_height, expected_width), 
                   num_classes=len(command_mapping))
    cnn.load('cnn_weights.npz')
    pred_idx = cnn.predict(x_input[0])
    reverse_mapping = {v: k for k, v in command_mapping.items()}
    print(f"\n🎧 Prediction: {reverse_mapping[pred_idx]}")
if __name__ == "__main__":
    print("\n🎤 Voice Command CNN Classifier")
    print("1. Train model")
    print("2. Test 'untitled.wav'")
    choice = input("\n👉 Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\n🚀 Starting model training...")
        X, y = process_all_files()
        
        # Calculate expected dimensions
        expected_height = N_MELS
        expected_width = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1
        
        # Resize all spectrograms to expected dimensions
        from scipy.ndimage import zoom
        X_reshaped = []
        for spec in X:
            if spec.shape != (expected_height, expected_width):
                zoom_factors = (expected_height/spec.shape[0], expected_width/spec.shape[1])
                spec = zoom(spec, zoom_factors)
            X_reshaped.append(spec[np.newaxis, :, :])  # Add channel dim
        
        X_reshaped = np.array(X_reshaped)
    
        cnn = SimpleCNN(input_shape=(expected_height, expected_width), 
                       num_classes=len(command_mapping))
        cnn.train(X_reshaped, y, epochs=10)
        cnn.save('cnn_weights.npz')
        print("\n💾 Model saved to 'cnn_weights.npz'")

    elif choice == "2":
        test_file = "untitled.wav"
        if os.path.exists(test_file):
            predict_wav_with_cnn(test_file)
        else:
            print(f"[ERROR] File '{test_file}' not found.")
    else:
        print("[ERROR] Invalid choice. Enter 1 or 2.")