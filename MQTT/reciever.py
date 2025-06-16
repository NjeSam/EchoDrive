import paho.mqtt.client as mqtt
import os

MQTT_BROKER_ZT_IP = "10.241.95.113"
MQTT_PORT = 1883
MQTT_TOPIC_AUDIO = "audio/files/new"
MQTT_CLIENT_ID = "audio_receiver_pc"
SAVE_DIRECTORY = "received_audio_files"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER_ZT_IP}")
        client.subscribe(MQTT_TOPIC_AUDIO, qos=1)
        print(f"Subscribed to topic: {MQTT_TOPIC_AUDIO}")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic} | Size: {len(msg.payload)} bytes")
    
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    filename = os.path.join(SAVE_DIRECTORY, "temp_file.wav")

    try:
        with open(filename, "wb") as f:
            f.write(msg.payload)
        print(f"Audio file saved as: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

def start_receiver():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER_ZT_IP, MQTT_PORT, 60)
        client.loop_forever()
    except ConnectionRefusedError:
        print(f"Connection refused. Is the MQTT broker at {MQTT_BROKER_ZT_IP}:{MQTT_PORT} running and accessible?")
    except KeyboardInterrupt:
        print("Receiver stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client.is_connected():
            client.disconnect()
        print("Disconnected from MQTT broker.")

if __name__ == "__main__":
    print(f"Starting audio receiver. Listening for messages on {MQTT_TOPIC_AUDIO}...")
    print(f"Received files will be saved in '{SAVE_DIRECTORY}' directory.")
    start_receiver()
