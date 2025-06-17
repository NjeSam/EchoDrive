import os
import librosa
import numpy as np

# 🧾 Ulazni folder sa .wav fajlovima
input_folder = 'C:/Users/ACER/Documents/GitHub/EchoDrive/Projekat/wav_commands'

# 📁 Izlazni folderi
output_root = 'spektrogrami_barve'
poznato_folder = os.path.join(output_root, 'poznato')
nepoznato_folder = os.path.join(output_root, 'nepoznato')

# 👤 Ključne reči u imenima
imena = ['DJORDJE', 'LAN', 'NJEGOS']

# 📂 Kreiranje foldera
for ime in imena:
    os.makedirs(os.path.join(poznato_folder, ime), exist_ok=True)
os.makedirs(nepoznato_folder, exist_ok=True)

# 🎵 Funkcija za pravljenje i čuvanje log-mel spektrograma
def sacuvaj_spektrogram_npy(wav_putanja, izlaz_putanja):
    try:
        y, sr = librosa.load(wav_putanja, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        np.save(izlaz_putanja, S_dB)
    except Exception as e:
        print(f'Greška kod fajla {wav_putanja}: {e}')

# 🚀 Glavna petlja
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.wav'):
        putanja_wav = os.path.join(input_folder, filename)

        ime_pronadjeno = None
        for ime in imena:
            if ime in filename.upper():
                ime_pronadjeno = ime
                break

        if ime_pronadjeno:
            izlaz_folder = os.path.join(poznato_folder, ime_pronadjeno)
        else:
            izlaz_folder = nepoznato_folder

        osnovno_ime = os.path.splitext(filename)[0]
        izlaz_putanja = os.path.join(izlaz_folder, osnovno_ime + '.npy')

        sacuvaj_spektrogram_npy(putanja_wav, izlaz_putanja)

print('✅ Gotovo: svi spektrogrami su sačuvani kao .npy fajlovi.')
