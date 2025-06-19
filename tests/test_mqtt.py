import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Dodaj korenski direktorij projekta v Python pot
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from MQTT import sender

@patch('MQTT.sender.sd.rec')
@patch('MQTT.sender.sd.wait')
@patch('MQTT.sender.mqtt.Client')
def test_sender_publishes_correct_data(mock_mqtt_client, mock_wait, mock_rec):
    """
    Integracijski test, ki preveri, ali funkcija za pošiljanje
    poskuša objaviti sporočilo na pravo temo.
    """
    # Ustvarimo lažen numpy array, ki ga funkcija 'write_wav' pričakuje.
    # Uporabimo dtype 'int16', kot je specificirano v originalni kodi.
    dummy_recording = np.zeros((44100, 1), dtype=np.int16)
    mock_rec.return_value = dummy_recording

    mock_instance = MagicMock()
    mock_mqtt_client.return_value = mock_instance
    
    sender.is_mqtt_connected = True
    sender.mqtt_client_instance = mock_instance

    sender.record_button = MagicMock()
    sender.status_label = MagicMock()
    sender.messagebox = MagicMock()

    # Sedaj bo ta klic uspel, ker 'write_wav' dobi veljavne podatke
    sender.record_and_send_audio_task()

    # Preverimo, ali je bila metoda 'publish' klicana točno enkrat
    mock_instance.publish.assert_called_once()
    
    # Preverimo argumente klica
    args, kwargs = mock_instance.publish.call_args
    assert 'payload' in kwargs
    assert 'qos' in kwargs
    assert kwargs['qos'] == 1
    assert isinstance(kwargs['payload'], bytes)
