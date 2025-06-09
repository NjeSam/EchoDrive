import tkinter as tk
from tkinter import ttk, messagebox
import paho.mqtt.client as mqtt
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import io
import threading
import time
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import sys

# --- Configuration ---
MQTT_BROKER_IP = "10.241.92.217"
MQTT_PORT = 1883
MQTT_TOPIC_AUDIO = "audio/files/new"
MQTT_CLIENT_ID = f"audio_sender_vm_{int(time.time())}"
PROMETHEUS_PORT_SENDER = 8000

RECORD_DURATION_SECONDS = 3
SAMPLE_RATE = 44100
CHANNELS = 1

# --- Prometheus Metrics ---
MESSAGES_SENT = Counter('sender_messages_sent_total', 'Total messages sent from the sender.')
BYTES_SENT = Counter('sender_bytes_sent_total', 'Total bytes sent from the sender.')
IS_MQTT_CONNECTED = Gauge('sender_mqtt_connected_status', 'Shows if the MQTT client is connected (1) or not (0).')
PROCESSING_TIME = Histogram('sender_processing_duration_seconds', 'Time spent recording and sending audio.')

# --- Global Variables ---
mqtt_client_instance = None
root = None

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties):
    print(f"--- on_connect callback with result code: {rc} ---")
    if rc == 0:
        print("[SUCCESS] Connected to MQTT Broker.")
        status_label.config(text=f"Connected to Broker: {MQTT_BROKER_IP}")
        IS_MQTT_CONNECTED.set(1)
        record_button.config(state=tk.NORMAL)
    else:
        error_map = {1: "incorrect protocol version", 2: "invalid client identifier", 3: "server unavailable", 4: "bad username/password", 5: "not authorised"}
        error_msg = error_map.get(rc, f"unknown error {rc}")
        print(f"[ERROR] Failed to connect: {error_msg}")
        status_label.config(text=f"Connection Failed: {error_msg}")
        IS_MQTT_CONNECTED.set(0)
        record_button.config(state=tk.DISABLED)

def on_publish(client, userdata, mid, rc, properties):
    print(f"--- on_publish callback for MID {mid} ---")

def on_disconnect(client, userdata, mid, rc, properties):
    print(f"--- Disconnected from MQTT with result code: {rc} ---")
    IS_MQTT_CONNECTED.set(0)
    status_label.config(text="Disconnected. Please restart.")
    record_button.config(state=tk.DISABLED)

# --- Core Application Logic ---
def record_and_send_audio_task():
    record_button.config(state=tk.DISABLED)
    status_label.config(text=f"Recording for {RECORD_DURATION_SECONDS}s...")
    try:
        with PROCESSING_TIME.time():
            recording = sd.rec(int(RECORD_DURATION_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
            sd.wait()
            status_label.config(text="Recording complete. Sending...")
            
            wav_bytes_io = io.BytesIO()
            write_wav(wav_bytes_io, SAMPLE_RATE, recording)
            audio_bytes = wav_bytes_io.getvalue()
            wav_bytes_io.close()

            if mqtt_client_instance and mqtt_client_instance.is_connected():
                result_tuple = mqtt_client_instance.publish(MQTT_TOPIC_AUDIO, payload=audio_bytes, qos=1)
                print(f"Publish call sent. Waiting for confirmation for MID: {result_tuple.mid}")
                result_tuple.wait_for_publish(timeout=5)
                
                if result_tuple.is_published():
                    status_label.config(text=f"Audio sent ({len(audio_bytes)} bytes). Ready for next.")
                    MESSAGES_SENT.inc()
                    BYTES_SENT.inc(len(audio_bytes))
                else:
                    status_label.config(text="Publish timed out. Message not confirmed.")
            else:
                status_label.config(text="Cannot send: Not connected.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        if mqtt_client_instance and mqtt_client_instance.is_connected():
            record_button.config(state=tk.NORMAL)
            print("Task finished. Button re-enabled.")

def start_record_and_send_thread():
    threading.Thread(target=record_and_send_audio_task, daemon=True).start()

def on_closing():
    print("Close button pressed. Shutting down cleanly.")
    if mqtt_client_instance:
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()
    root.destroy()
    print("Shutdown complete.")

def main():
    global root, record_button, status_label, mqtt_client_instance
    
    print(f"Starting Prometheus metrics server on http://localhost:{PROMETHEUS_PORT_SENDER}")
    start_http_server(PROMETHEUS_PORT_SENDER)
    IS_MQTT_CONNECTED.set(0)
    
    root = tk.Tk()
    root.title("Audio Sender")
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill=tk.BOTH)
    
    record_button = ttk.Button(main_frame, text="Record & Send", command=start_record_and_send_thread)
    record_button.pack(pady=20)
    record_button.config(state=tk.DISABLED)
    
    status_label = ttk.Label(main_frame, text="Initializing...", wraplength=400, justify=tk.CENTER)
    status_label.pack(pady=10)
    
    try:
        print(f"Initializing MQTT client to connect to {MQTT_BROKER_IP}:{MQTT_PORT}...")
        mqtt_client_instance = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
        mqtt_client_instance.on_connect = on_connect
        mqtt_client_instance.on_publish = on_publish
        mqtt_client_instance.on_disconnect = on_disconnect
        
        status_label.config(text=f"Connecting to {MQTT_BROKER_IP}...")
        mqtt_client_instance.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
        mqtt_client_instance.loop_start()
    except Exception as e:
        messagebox.showerror("Connection Error", f"Could not connect to broker at {MQTT_BROKER_IP}.\nError: {e}")
        root.destroy()
        sys.exit(1)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
