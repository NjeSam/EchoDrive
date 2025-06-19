import numpy as np
import os
import sys

# --- POPRAVEK: Dodaj korenski direktorij projekta v Python pot ---
# To omogoča uvoz modulov iz mape 'Projekat'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# -----------------------------------------------------------------

# Sedaj lahko uvozimo funkcije iz mape 'Projekat'
from Projekat.neuronska_projekat import create_spectrogram, extract_command

# Definicija konstant za testiranje
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
HOP_LENGTH = 512

def test_extract_command_from_filename():
    """
    Testira, ali funkcija extract_command pravilno prepozna ukaz
    iz podanega imena datoteke.
    """
    # Testni primer 1: Preprost ukaz
    filename1 = "TURN ON THE RADIO DJORDJE 1.wav"
    assert extract_command(filename1) == "TURN ON THE RADIO"

    # Testni primer 2: Ukaz z več besedami
    filename2 = "TURN OFF THE AIR CONDITIONER LAN.wav"
    assert extract_command(filename2) == "TURN OFF THE AIR CONDITIONER"
    
    # Testni primer 3: Ime datoteke brez veljavnega ukaza
    filename3 = "random_sound.wav"
    assert extract_command(filename3) is None

def test_create_spectrogram_shape():
    """
    Testira, ali funkcija create_spectrogram ustvari numpy matriko
    s pričakovanimi dimenzijami.
    """
    # Ustvari testni avdio signal dolžine 3 sekunde
    dummy_audio = np.random.randn(SAMPLE_RATE * DURATION)
    
    # Generiraj spektrogram
    spectrogram = create_spectrogram(dummy_audio, SAMPLE_RATE)
    
    # Izračunaj pričakovano obliko
    expected_columns = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1
    expected_shape = (N_MELS, expected_columns)
    
    # Preveri, ali ima spektrogram pravilno obliko
    assert spectrogram.shape == expected_shape, "Oblika spektrograma je napačna"
    
    # Preveri, ali je izhod numpy matrika
    assert isinstance(spectrogram, np.ndarray), "Izhod bi moral biti numpy matrika"
