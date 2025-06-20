import os
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog, ttk
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
from PIL import Image, ImageTk, ImageSequence


# Nastavitve
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3
expected_shape = (N_MELS, int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1)

MODEL_PATH = "govornik_model.h5"
CONFIDENCE_THRESHOLD = 0.97
MARGIN_THRESHOLD = 0.2
LABELS = ['DJORDJE', 'LAN', 'NJEGOS']
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

# Globalna spremenljivka za AI stanje
ai_enabled = False

# Mape ukazov
command_mapping = {
    'AI ON': 0, 'AI OFF': 1, 'TURN ON THE AIR CONDITIONER': 2, 'TURN OFF THE AIR CONDITIONER': 3,
    'TURN ON THE RADIO': 4, 'TURN OFF THE RADIO': 5, 'SWITCH THE RADIO STATION': 6,
    'MUTE THE RADIO': 7, 'TURN UP THE VOLUME': 8, 'TURN DOWN THE VOLUME': 9,
    'TURN ON THE NAVIGATION': 10, 'TURN OFF THE NAVIGATION': 11
}
inv_map = {v: k for k, v in command_mapping.items()}

# Spotify nastavitev
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="d5b6a52f2b1b47288972ddf296b46ccd",
    client_secret="b68ef098e4b1457d9fac6d5a79ec4ce7",
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

# Funkcije za glasbo
def play_music():
    sp.start_playback()

def pause_music():
    sp.pause_playback()

def next_track():
    sp.next_track()

def volume_up():
    current = sp.current_playback()
    if current and 'device' in current:
        sp.volume(min(100, current['device']['volume_percent'] + 15))

def volume_down():
    current = sp.current_playback()
    if current and 'device' in current:
        sp.volume(max(0, current['device']['volume_percent'] - 15))

def mute():
    current = sp.current_playback()
    if current and 'device' in current:
        sp.volume(0)

# Funkcija za spektrogram
def create_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())
    return norm_S

# Prepoznavanje ukaza
def recognize_command(filepath):
    global ai_enabled
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
    spec = create_spectrogram(y, sr)
    if spec.shape != expected_shape:
        zoom_factors = (expected_shape[0] / spec.shape[0], expected_shape[1] / spec.shape[1])
        spec = zoom(spec, zoom_factors)
    x_input = spec[np.newaxis, ..., np.newaxis]
    model = load_model("keras_model.h5")
    pred = np.argmax(model.predict(x_input))
    command = inv_map[pred]

    # AI status
    if command == "AI ON":
        ai_enabled = True
    elif command == "AI OFF":
        ai_enabled = False

    # Izvedba glasbenih ukazov samo, če je AI ON
    if ai_enabled:
        if command == "TURN ON THE RADIO":
            play_music()
        elif command == "TURN OFF THE RADIO":
            pause_music()
        elif command == "SWITCH THE RADIO STATION":
            next_track()
        elif command == "TURN UP THE VOLUME":
            volume_up()
        elif command == "TURN DOWN THE VOLUME":
            volume_down()
        elif command == "MUTE THE RADIO":
            mute()
        elif command == "TURN ON THE NAVIGATION":
            webbrowser.open("https://www.google.com/maps")


    return command


# Funkcija za generisanje spektrograma
def create_spectrogram2(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < DURATION * sr:
        y = np.pad(y, (0, DURATION * sr - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db / 80.0 + 1  # normalizacija u [0,1]
    return mel_db.astype(np.float32)


def predict_speaker(file_path):
    if not os.path.exists(MODEL_PATH):
        print("❌ Nema modela")
        return

    model = load_model(MODEL_PATH)
    spec = create_spectrogram2(file_path)
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
    return label

# Grafični uporabniški vmesnik
class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Command Recognizer")
        self.root.geometry("500x500")
        self.root.configure(bg="#f0f0f0")

        self.bg_image = Image.open("background.jpg") 
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.button = tk.Button(root, text="Izberi WAV datoteko", command=self.load_file, font=("Arial", 12))
        self.button.pack(pady=10)

        # Okvir za komando in govorca
        self.command_frame = tk.LabelFrame(root, text="Prepoznana Komanda", font=("Arial", 12), bg="#ffffff", width=400, height=60)
        self.command_frame.pack(pady=5)
        self.command_label = tk.Label(self.command_frame, text="Ni komande", font=("Arial", 14), bg="#ffffff")
        self.command_label.pack(pady=5)

        self.speaker_frame = tk.LabelFrame(root, text="Prepoznan Govorec", font=("Arial", 12), bg="#ffffff", width=400, height=60)
        self.speaker_frame.pack(pady=5)
        self.speaker_label = tk.Label(self.speaker_frame, text="Ni govorca", font=("Arial", 14), bg="#ffffff")
        self.speaker_label.pack(pady=5)

        self.status = tk.Label(root, text=f"AI status: {ai_enabled}", font=("Arial", 10), fg="green" if ai_enabled else "red", bg="#f0f0f0")
        self.status.pack(pady=10)

        # GIF prikaz
        self.gif_label = tk.Label(root, bg="#f0f0f0")
        self.gif_label.pack(pady=10)
        self.current_gif = None
        self.update_gif("gifs/air_off.gif")

       

    def update_gif(self, gif_path):
        if not os.path.exists(gif_path):
            self.gif_label.config(image='', text="(Ni GIF datoteke)")
            return

        # Naloži GIF in ga predvaja kot animacijo
        self.frames = [ImageTk.PhotoImage(img.copy().resize((320, 320))) for img in ImageSequence.Iterator(Image.open(gif_path))]
        self.gif_index = 0

        def update_frame():
            if hasattr(self, 'frames') and self.frames:
                self.gif_label.config(image=self.frames[self.gif_index])
                self.gif_index = (self.gif_index + 1) % len(self.frames)
                self.root.after(100, update_frame)

        update_frame()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filepath:
            command = recognize_command(filepath)
            label = predict_speaker(filepath)

            emoji_map = {
                "TURN ON THE AIR CONDITIONER": "❄️",
                "TURN OFF THE AIR CONDITIONER": "🔥",
                "TURN ON THE RADIO": "📻",
                "MUTE THE RADIO": "🔇",
                "TURN UP THE VOLUME": "🔊",
                "TURN DOWN THE VOLUME": "🔉",
                "TURN ON THE NAVIGATION": "🗺️",
            }
            emoji = emoji_map.get(command, "")

            self.command_label.config(text=f"{emoji} {command}")
            self.speaker_label.config(text=f"🎤 {label}")
            self.status.config(text=f"AI status: {ai_enabled}", fg="green" if ai_enabled else "red")

            # Posodobi GIF
            if command == "TURN ON THE AIR CONDITIONER":
                self.update_gif("gifs/air_on.gif")
            elif command == "TURN OFF THE AIR CONDITIONER":
                self.update_gif("gifs/air_off.gif")
            else:
                self.gif_label.config(image='')  # prazno            

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGUI(root)
    root.mainloop()
