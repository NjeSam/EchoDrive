import tkinter as tk
from tkinter import ttk, messagebox
import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import io
import threading

MQTT_BROKER_ZT_IP = "10.241.95.113"
MQTT_PORT = 1883
MQTT_TOPIC_AUDIO = "audio/files/new"
MQTT_CLIENT_ID = "audio_sender_vm_gui"

RECORD_DURATION_SECONDS = 3
SAMPLE_RATE = 44100
CHANNELS = 1

mqtt_client_instance = None
is_mqtt_connected = False

def on_connect(client, userdata, flags, rc):
    global is_mqtt_connected
    print(f"--- DEBUG: on_connect called with rc: {rc}, flags: {flags} ---")
    if rc == 0:
        status_label.config(text=f"Connected to MQTT: {MQTT_BROKER_ZT_IP}")
        print(f"Connected to MQTT Broker: {MQTT_BROKER_ZT_IP}")
        is_mqtt_connected = True
    else:
        status_label.config(text=f"Connection failed, rc: {rc}")
        print(f"Failed to connect, return code {rc}")
        is_mqtt_connected = False

def on_publish(client, userdata, mid):
    status_label.config(text=f"Audio published (MID: {mid})")
    print(f"Audio published with MID {mid}")

def on_disconnect(client, userdata, rc):
    global is_mqtt_connected
    print(f"--- DEBUG: on_disconnect called with rc: {rc} ---")
    status_label.config(text=f"Disconnected from MQTT (rc: {rc})")
    print(f"Disconnected from MQTT broker, return code: {rc}")
    is_mqtt_connected = False
    if rc != 0:
        print("Unexpected disconnection.")

def record_and_send_audio_task():
    global mqtt_client_instance, is_mqtt_connected
    record_button.config(state=tk.DISABLED)
    status_label.config(text=f"Recording for {RECORD_DURATION_SECONDS} seconds...")
    print(f"Recording for {RECORD_DURATION_SECONDS} seconds...")
    try:
        recording = sd.rec(int(RECORD_DURATION_SECONDS * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=CHANNELS,
                           dtype='int16')
        sd.wait()
        status_label.config(text="Recording complete. Preparing to send...")
        print("Recording complete.")
        wav_bytes_io = io.BytesIO()
        write_wav(wav_bytes_io, SAMPLE_RATE, recording)
        audio_bytes = wav_bytes_io.getvalue()
        wav_bytes_io.close()
        print(f"Audio data size: {len(audio_bytes)} bytes")

        if mqtt_client_instance and is_mqtt_connected:
            status_label.config(text=f"Sending {len(audio_bytes)} bytes...")
            MQTT_PUB_SUCCESS = 0
            result_tuple = mqtt_client_instance.publish(MQTT_TOPIC_AUDIO, payload=audio_bytes, qos=1)
            if result_tuple[0] == MQTT_PUB_SUCCESS:
                print(f"Publish call successful, MID={result_tuple[1]}")
            else:
                status_label.config(text=f"Publish failed, rc: {result_tuple[0]}")
                print(f"Publish call failed, rc: {result_tuple[0]}")
        else:
            status_label.config(text="MQTT client not connected (flag). Cannot send.")
            print(f"MQTT client not connected based on is_mqtt_connected flag: {is_mqtt_connected}")
            if mqtt_client_instance and hasattr(mqtt_client_instance, '_state'):
                 print(f"Internal client state _state: {mqtt_client_instance._state}")
            messagebox.showerror("MQTT Error", "Not connected to MQTT broker.")
            
    except Exception as e:
        status_label.config(text=f"Error: {e}")
        print(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        record_button.config(state=tk.NORMAL)

def start_record_and_send_thread():
    thread = threading.Thread(target=record_and_send_audio_task, daemon=True)
    thread.start()

def setup_mqtt_client():
    global mqtt_client_instance
    try:
        mqtt_client_instance = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True, protocol=mqtt.MQTTv311)
        print(f"Initialized paho.mqtt.client.Client for paho-mqtt 1.x style.")

        mqtt_client_instance.on_connect = on_connect
        mqtt_client_instance.on_publish = on_publish
        mqtt_client_instance.on_disconnect = on_disconnect

        status_label.config(text=f"Connecting to {MQTT_BROKER_ZT_IP}...")
        mqtt_client_instance.connect_async(MQTT_BROKER_ZT_IP, MQTT_PORT, 60)
        mqtt_client_instance.loop_start()
    except Exception as e:
        status_label.config(text=f"MQTT Setup Error: {e}")
        print(f"MQTT Setup Error: {e}")
        messagebox.showerror("MQTT Error", f"Could not initialize MQTT client: {e}")

def create_gui():
    global record_button, status_label
    root = tk.Tk()
    root.title("Audio Recorder Sender")
    root.geometry("400x200")
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 10))
    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.pack(expand=True, fill=tk.BOTH)
    record_button = ttk.Button(main_frame, text="Record & Send (3s)", command=start_record_and_send_thread)
    record_button.pack(pady=20)
    status_label = ttk.Label(main_frame, text="Press button to record and send.", wraplength=350, justify=tk.CENTER)
    status_label.pack(pady=10)
    setup_mqtt_client()
    
    def on_closing():
        global is_mqtt_connected
        if mqtt_client_instance:
            print("Disconnecting MQTT client...")
            is_mqtt_connected = False 
            mqtt_client_instance.loop_stop(force=False)
            try:
                mqtt_client_instance.disconnect()
            except Exception as e_disc:
                 print(f"Exception during disconnect on closing: {e_disc}")
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    create_gui()
