import numpy as np
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Dodaj korenski direktorij projekta v Python pot
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Uvozi funkcijo iz skripta za prepoznavanje govorca
from Projekat.neuronska_barve import predict_speaker

# Preskoči test, če TensorFlow ni na voljo
try:
    import tensorflow
except ImportError:
    tensorflow = None

@unittest.skipIf(tensorflow is None, "TensorFlow ni nameščen")
@patch('Projekat.neuronska_barve.os.path.exists')  # <-- FIX 1: Mock os.path.exists
@patch('Projekat.neuronska_barve.load_model')
def test_predict_speaker_logic(mock_load_model, mock_path_exists):
    """
    Testira logiko odločanja v funkciji predict_speaker.
    """
    # Povemo mocku, naj vedno vrne True, da preskoči preverjanje datoteke
    mock_path_exists.return_value = True

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    fake_file_path = "fake_audio.wav"

    # Pripravimo mock za create_spectrogram, ki ga klice predict_speaker
    with patch('Projekat.neuronska_barve.create_spectrogram', return_value=np.zeros((128, 130))):
        
        # Primer 1: Visoko zaupanje -> DJORDJE
        mock_model.predict.return_value = np.array([[0.98, 0.01, 0.01]])
        with patch('builtins.print') as mock_print:
            predict_speaker(fake_file_path)
            # Preverimo, ali je kateri od klicev print vseboval 'DJORDJE'
            assert any('DJORDJE' in call.args[0] for call in mock_print.call_args_list)

        # Primer 2: Nizko zaupanje -> NEPOZNATO
        mock_model.predict.return_value = np.array([[0.5, 0.3, 0.2]])
        with patch('builtins.print') as mock_print:
            predict_speaker(fake_file_path)
            assert any('NEPOZNATO' in call.args[0] for call in mock_print.call_args_list)

        # Primer 3: Majhna razlika -> NEPOZNATO
        mock_model.predict.return_value = np.array([[0.7, 0.65, 0.05]])
        with patch('builtins.print') as mock_print:
            predict_speaker(fake_file_path)
            assert any('NEPOZNATO' in call.args[0] for call in mock_print.call_args_list)
