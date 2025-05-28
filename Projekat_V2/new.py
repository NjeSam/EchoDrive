print("DEBUG: Top of new.py")
import os
print("DEBUG: After import os")
import numpy as np
print("DEBUG: After import numpy")
import torch
print("DEBUG: After import torch")
import torch.nn as nn
print("DEBUG: After import torch.nn")
import torch.optim as optim
print("DEBUG: After import torch.optim")
from torch.utils.data import Dataset, DataLoader, random_split
print("DEBUG: After import torch.utils.data")
import librosa
print("DEBUG: After import librosa")
import librosa.display
print("DEBUG: After import librosa.display")
import sounddevice as sd
print("DEBUG: After import sounddevice")
import scipy.io.wavfile as wav
print("DEBUG: After import scipy.io.wavfile")
import tempfile
print("DEBUG: After import tempfile")
import threading
print("DEBUG: After import threading")
import queue # For more complex GUI updates if needed, not used heavily here
print("DEBUG: After import queue")

import customtkinter as ctk
print("DEBUG: After import customtkinter")
from tkinter import filedialog, messagebox # Keep messagebox for critical errors/info
print("DEBUG: After import tkinter filedialog, messagebox")
import matplotlib.pyplot as plt
print("DEBUG: After import matplotlib.pyplot")
import traceback # Moved this import higher for earlier availability
print("DEBUG: After import traceback")
from typing import Union, Optional # Optional is also good here
print("DEBUG: After import typing Union, Optional")

# --- Configuration ---
print("DEBUG: Before CONFIG definition")
CONFIG = {
    "dataset_dir": 'wav_commands_augmented',
    "model_path": 'command_cnn.pth',
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "num_epochs": 10, # Reduced for faster demo, was 30
    "train_val_split": 0.8,
    "record_seconds": 2.5,
    "expected_audio_length_samples": 16000,
}
print("DEBUG: After CONFIG definition")

# --- Utility: extract command name before first underscore ---
print("DEBUG: Before parse_command definition")
def parse_command(filename: str) -> Union[str, None]:  # <--- Potential issue here with type hint
    base = os.path.basename(filename)
    if '_' in base:
        cmd = base.split('_')[0].strip()
        return cmd
    print(f"Warning: File '{filename}' does not follow 'command_extra.wav' format. Skipping.")
    return None
print("DEBUG: After parse_command definition")

# --- Dataset Class ---
print("DEBUG: Before CommandDataset class definition")
class CommandDataset(Dataset):
    def __init__(self, root_dir: str, sr: int, expected_len_samples: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        print(f"DEBUG_DATASET: __init__ called for {root_dir}")
        self.root_dir = root_dir
        self.sr = sr
        self.expected_len_samples = expected_len_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.commands_list = []
        self.filepaths = []
        self.labels = []

        if not os.path.isdir(self.root_dir):
            print(f"DEBUG_DATASET_ERROR: Dataset directory '{self.root_dir}' not found.")
            self.num_classes = 0
            return

        for fname in sorted(os.listdir(self.root_dir)):
            if not fname.lower().endswith('.wav'):
                continue
            cmd = parse_command(fname)
            if not cmd:
                continue
            
            if cmd not in self.commands_list:
                self.commands_list.append(cmd)
            
            self.filepaths.append(os.path.join(self.root_dir, fname))
            self.labels.append(self.commands_list.index(cmd))

        if not self.commands_list:
            print("DEBUG_DATASET: No valid commands found.")
        else:
            print(f"DEBUG_DATASET: Found commands: {self.commands_list}")
            print(f"DEBUG_DATASET: Total samples: {len(self.filepaths)}")
        
        self.num_classes = len(self.commands_list)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr_loaded = librosa.load(path, sr=self.sr)
        except Exception as e:
            print(f"DEBUG_DATASET_ERROR: Error loading {path}: {e}")
            return torch.zeros(1, self.n_mels, int(np.ceil(self.expected_len_samples / self.hop_length))), -1 

        if len(waveform) < self.expected_len_samples:
            waveform = np.pad(waveform, (0, self.expected_len_samples - len(waveform)), mode='constant')
        else:
            waveform = waveform[:self.expected_len_samples]
        
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sr,
                                                  n_fft=self.n_fft, hop_length=self.hop_length,
                                                  n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mean = mel_db.mean()
        std = mel_db.std()
        mel_norm = (mel_db - mean) / (std + 1e-9)
        
        tensor = torch.tensor(mel_norm).unsqueeze(0).float()
        return tensor, label
print("DEBUG: After CommandDataset class definition")

# --- CNN Model ---
print("DEBUG: Before CNN class definition")
class CNN(nn.Module):
    def __init__(self, num_classes: int, n_mels: int, time_frames: int):
        super().__init__()
        print(f"DEBUG_CNN: __init__ called. num_classes={num_classes}, n_mels={n_mels}, time_frames={time_frames}")
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        dummy_input = torch.zeros(1, 1, n_mels, time_frames)
        dummy_output = self.features(dummy_input)
        feat_size = dummy_output.numel()
        print(f"DEBUG_CNN: Calculated feat_size: {feat_size}")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
print("DEBUG: After CNN class definition")

# --- Training Logic (to be run in a thread) ---
print("DEBUG: Before run_training_logic definition")
def run_training_logic(app_ref, config_param): # Use config_param
    print("DEBUG_TRAINING: run_training_logic started.")
    try:
        app_ref.update_status("Starting training...")
        if not os.path.exists(config_param["dataset_dir"]) or not os.listdir(config_param["dataset_dir"]):
            error_msg = f"Dataset directory '{config_param['dataset_dir']}' is missing or empty."
            app_ref.update_status(error_msg)
            if hasattr(app_ref, 'after'):
                 app_ref.after(0, lambda: messagebox.showerror("Training Error", f"{error_msg}\nPlease create it and add WAV files."))
            else:
                 print(f"ERROR: {error_msg}")
            app_ref.training_finished()
            return

        dataset = CommandDataset(
            root_dir=config_param["dataset_dir"],
            sr=config_param["sample_rate"],
            expected_len_samples=config_param["expected_audio_length_samples"],
            n_fft=config_param["n_fft"],
            hop_length=config_param["hop_length"],
            n_mels=config_param["n_mels"]
        )

        if dataset.num_classes == 0:
            error_msg = "No valid command data found. Training aborted."
            app_ref.update_status(error_msg)
            if hasattr(app_ref, 'after'):
                app_ref.after(0, lambda: messagebox.showerror("Training Error", "No valid command data found in dataset. Check file names and directory."))
            else:
                print(f"ERROR: {error_msg}")
            app_ref.training_finished()
            return

        train_size = int(config_param["train_val_split"] * len(dataset))
        val_size = len(dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            error_msg = "Dataset too small for train/validation split. Need more diverse samples."
            app_ref.update_status(error_msg)
            if hasattr(app_ref, 'after'):
                app_ref.after(0, lambda: messagebox.showerror("Training Error", "Dataset too small. Please add more data."))
            else:
                print(f"ERROR: {error_msg}")
            app_ref.training_finished()
            return

        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=config_param["batch_size"], shuffle=True, num_workers=0) 
        val_loader = DataLoader(val_ds, batch_size=config_param["batch_size"], num_workers=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        app_ref.update_status(f"Using device: {device}")

        time_frames = int(np.ceil(config_param["expected_audio_length_samples"] / config_param["hop_length"]))
        model = CNN(dataset.num_classes, config_param["n_mels"], time_frames).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config_param["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accuracies = []
        
        app_ref.update_log_area("Training Log:\n")

        for epoch in range(1, config_param["num_epochs"] + 1):
            model.train()
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                if isinstance(y, torch.Tensor) and y.numel() > 0 and y[0] == -1: 
                    app_ref.update_log_area(f"Skipping problematic batch {batch_idx} in epoch {epoch}\n")
                    continue
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_losses.append(avg_train_loss)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader: # Renamed to avoid conflict
                    if isinstance(y_val, torch.Tensor) and y_val.numel() > 0 and y_val[0] == -1: continue
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    outputs = model(x_val)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_val.size(0)
                    correct += (predicted == y_val).sum().item()
            
            val_acc = correct / total if total > 0 else 0
            val_accuracies.append(val_acc)
            
            status_msg = f"Epoch {epoch}/{config_param['num_epochs']} — Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}"
            app_ref.update_status(status_msg)
            app_ref.update_log_area(status_msg + "\n")
            app_ref.update_progressbar(epoch / config_param["num_epochs"])

        torch.save({
            'model_state_dict': model.state_dict(),
            'commands': dataset.commands_list,
            'n_mels': config_param['n_mels'],
            'time_frames': time_frames 
        }, config_param["model_path"])
        
        app_ref.update_status(f"Training complete. Model saved to {config_param['model_path']}")
        app_ref.update_log_area(f"Training complete. Model saved to {config_param['model_path']}\n")
        
        if hasattr(app_ref, 'after'):
             app_ref.after(0, lambda: plot_training_history(train_losses, val_accuracies, config_param["num_epochs"]))
        else:
             plot_training_history(train_losses, val_accuracies, config_param["num_epochs"])
        
        app_ref.load_model() # Reload model in app

    except Exception as e:
        error_msg = f"Training failed: {e}"
        print(f"DEBUG_TRAINING_ERROR: {error_msg}")
        traceback.print_exc()
        app_ref.update_status(error_msg)
        app_ref.update_log_area(error_msg + "\n")
        if hasattr(app_ref, 'after'):
            app_ref.after(0, lambda: messagebox.showerror("Training Error", error_msg))
        else:
            print(f"ERROR: {error_msg}")
    finally:
        app_ref.training_finished()
    print("DEBUG_TRAINING: run_training_logic finished.")
print("DEBUG: After run_training_logic definition")

# --- Plotting Function ---
print("DEBUG: Before plot_training_history definition")
def plot_training_history(train_losses, val_accuracies, num_epochs):
    print(f"DEBUG_PLOT: plot_training_history called. Epochs: {num_epochs}")
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("DEBUG_PLOT: plot_training_history finished, plt.show() called.")
print("DEBUG: After plot_training_history definition")

# --- Main Application Class ---
print("DEBUG: Before CommandApp class definition")
class CommandApp(ctk.CTk):
    def __init__(self, app_config): # Changed parameter name
        super().__init__()
        print("DEBUG_APP: CommandApp __init__ started.")
        self.config = app_config # Use the passed config
        self.model = None
        self.commands = []
        self.model_n_mels = 0
        self.model_time_frames = 0

        self.title("Voice Command Recognizer")
        self.geometry("700x550")
        # Appearance mode and theme are set once in __main__ before app creation.

        self._setup_ui()
        self.load_model() 
        print("DEBUG_APP: CommandApp __init__ finished.")

    def _setup_ui(self):
        print("DEBUG_APP: _setup_ui called")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) 
        self.grid_rowconfigure(2, weight=0) 

        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure((0,1,2,3), weight=0) # Adjusted column configure for theme
        controls_frame.grid_columnconfigure(0, weight=1) # Give button more space if needed
        controls_frame.grid_columnconfigure(1, weight=1)


        self.train_button = ctk.CTkButton(controls_frame, text="Train Model", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.record_button = ctk.CTkButton(controls_frame, text="Record & Predict", command=self.start_record_and_predict, state="disabled")
        self.record_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.show_spectrogram_var = ctk.BooleanVar(value=False) # Use ctk.BooleanVar
        self.show_spectrogram_checkbox = ctk.CTkCheckBox(controls_frame, text="Show Spectrogram", variable=self.show_spectrogram_var)
        self.show_spectrogram_checkbox.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.log_textbox = ctk.CTkTextbox(self, height=200, wrap="word")
        self.log_textbox.grid(row=1, column=0, padx=10, pady=(0,5), sticky="nsew")
        self.log_textbox.insert("end", "Welcome to Voice Command Recognizer!\n")
        self.log_textbox.configure(state="disabled") 

        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", mode="determinate")
        self.progress_bar.grid(row=2, column=0, padx=10, pady=(0,5), sticky="ew")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self, text="Status: Idle", anchor="w")
        self.status_label.grid(row=3, column=0, padx=10, pady=(0,10), sticky="ew")
        
        theme_frame = ctk.CTkFrame(controls_frame) 
        theme_frame.grid(row=0, column=3, padx=10, pady=5, sticky="e") # Use column 3
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left", padx=(0,5))
        self.theme_menu = ctk.CTkOptionMenu(theme_frame, values=["Light", "Dark", "System"],
                                            command=self.change_appearance_mode)
        self.theme_menu.pack(side="left")
        if hasattr(ctk, 'get_appearance_mode'):
             self.theme_menu.set(ctk.get_appearance_mode())
        else:
             self.theme_menu.set("System") 
        print("DEBUG_APP: _setup_ui finished")

    def change_appearance_mode(self, new_mode: str):
        print(f"DEBUG_APP: Changing appearance mode to {new_mode}")
        ctk.set_appearance_mode(new_mode)

    def update_status(self, message: str):
        self.after(0, lambda: self.status_label.configure(text=f"Status: {message}"))

    def update_log_area(self, message: str):
        def _update():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", message)
            self.log_textbox.see("end") 
            self.log_textbox.configure(state="disabled")
        self.after(0, _update)

    def update_progressbar(self, value: float):
        self.after(0, lambda: self.progress_bar.set(value))

    def load_model(self):
        print("DEBUG_APP: load_model called")
        self.update_status(f"Attempting to load model from {self.config['model_path']}...")
        if not os.path.isfile(self.config['model_path']):
            self.update_status(f"No model found at {self.config['model_path']}. Please train a model.")
            self.toggle_recording_button(enable=False)
            self.model = None
            self.commands = []
            return

        try:
            checkpoint = torch.load(self.config['model_path'], map_location='cpu')
            self.commands = checkpoint['commands']
            
            self.model_n_mels = checkpoint.get('n_mels', self.config['n_mels']) 
            self.model_time_frames = checkpoint.get('time_frames', int(np.ceil(self.config["expected_audio_length_samples"] / self.config["hop_length"])))

            self.model = CNN(num_classes=len(self.commands), n_mels=self.model_n_mels, time_frames=self.model_time_frames)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.update_status(f"Model loaded successfully. Commands: {self.commands}")
            self.update_log_area(f"Model loaded. Ready to recognize: {', '.join(self.commands)}\n")
            self.toggle_recording_button(enable=True)
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(f"DEBUG_APP_ERROR: {error_msg}")
            traceback.print_exc()
            self.update_status(error_msg)
            self.update_log_area(f"{error_msg}\n")
            self.after(0, lambda: messagebox.showerror("Model Load Error", f"Failed to load model: {e}"))
            self.model = None
            self.commands = []
            self.toggle_recording_button(enable=False)
        print("DEBUG_APP: load_model finished")

    def toggle_recording_button(self, enable: bool):
        self.record_button.configure(state="normal" if enable else "disabled")

    def training_finished(self):
        print("DEBUG_APP: training_finished called")
        self.after(0, lambda: self.train_button.configure(state="normal"))
        self.after(0, lambda: self.progress_bar.set(0)) 

    def start_training(self):
        print("DEBUG_APP: start_training called")
        self.train_button.configure(state="disabled")
        self.update_status("Initializing training...")
        self.update_progressbar(0)
        # Pass self.config to the training thread
        train_thread = threading.Thread(target=run_training_logic, args=(self, self.config), daemon=True)
        train_thread.start()

    def start_record_and_predict(self):
        print("DEBUG_APP: start_record_and_predict called")
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please train a model first.")
            return

        self.record_button.configure(state="disabled")
        self.update_status("Preparing to record...")
        
        record_thread = threading.Thread(target=self._record_and_predict_thread, daemon=True)
        record_thread.start()

    def _process_recorded_audio(self, waveform_full: np.ndarray) -> tuple[torch.Tensor | None, np.ndarray | None, np.ndarray | None]:
        print("DEBUG_APP: _process_recorded_audio called")
        if len(waveform_full) < self.config["expected_audio_length_samples"]:
            waveform = np.pad(waveform_full, (0, self.config["expected_audio_length_samples"] - len(waveform_full)), mode='constant')
        else:
            waveform = waveform_full[:self.config["expected_audio_length_samples"]]

        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.config["sample_rate"],
                                                  n_fft=self.config["n_fft"],
                                                  hop_length=self.config["hop_length"],
                                                  n_mels=self.model_n_mels) 
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mean = mel_db.mean()
        std = mel_db.std()
        mel_norm = (mel_db - mean) / (std + 1e-9)
        
        input_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float()
        return input_tensor, mel_norm, waveform

    def _record_and_predict_thread(self):
        try:
            self.update_status(f"Recording for {self.config['record_seconds']:.1f} seconds...")
            print(f"DEBUG_THREAD: Recording for {self.config['record_seconds']:.1f} seconds...") # Console log

            recording = sd.rec(int(self.config['record_seconds'] * self.config['sample_rate']),
                            samplerate=self.config['sample_rate'],
                            channels=1, dtype='float32')
            sd.wait()
            print("DEBUG_THREAD: Recording finished.") # Console log

            self.update_status("Processing audio...")
            recorded_waveform = recording.flatten()
            print(f"DEBUG_THREAD: Raw recorded waveform length: {len(recorded_waveform)}") # Console log

            # Ensure _process_recorded_audio returns the 1s waveform for saving
            input_tensor, mel_spectrogram_for_display, processed_1s_waveform = self._process_recorded_audio(recorded_waveform)
            print("DEBUG_THREAD: Audio processed.") # Console log

            if input_tensor is None:
                self.update_status("Audio processing failed.")
                print("ERROR_THREAD: Audio processing returned None tensor.") # Console log
                # self.after(0, lambda: self.record_button.configure(state="normal")) # Moved to finally
                return

            print("DEBUG_THREAD: Making prediction...") # Console log
            with torch.no_grad():
                prediction = self.model(input_tensor)
                predicted_idx = torch.argmax(prediction, dim=1).item()
            
            predicted_command = "Unknown" # Default
            if 0 <= predicted_idx < len(self.commands):
                predicted_command = self.commands[predicted_idx]
            else:
                print(f"ERROR_THREAD: Predicted index {predicted_idx} out of bounds for commands list (len {len(self.commands)})")


            result_message = f"Detected: {predicted_command}"
            print(f"DEBUG_THREAD: {result_message}") # Crucial console log

            # Schedule GUI updates on the main thread
            self.after(0, lambda msg=result_message: self.update_status(msg))
            self.after(0, lambda msg=f"{result_message}\n": self.update_log_area(msg))
            
            # --- Debug Audio Saving ---
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # More unique timestamp
                # Save the full recording
                full_rec_path = os.path.join(tempfile.gettempdir(), f"debug_full_rec_{predicted_command.replace(' ','_')}_{timestamp}.wav")
                wav.write(full_rec_path, self.config["sample_rate"], (recorded_waveform * 32767).astype(np.int16))
                print(f"DEBUG_THREAD: Saved full recording to {full_rec_path}")

                if processed_1s_waveform is not None:
                    seg_rec_path = os.path.join(tempfile.gettempdir(), f"debug_segment_rec_{predicted_command.replace(' ','_')}_{timestamp}.wav")
                    wav.write(seg_rec_path, self.config["sample_rate"], (processed_1s_waveform * 32767).astype(np.int16))
                    print(f"DEBUG_THREAD: Saved 1s segment to {seg_rec_path}")
            except Exception as e_save:
                print(f"ERROR_THREAD: Error saving debug audio: {e_save}")
            # --- End Debug Audio Saving ---

            if self.show_spectrogram_var.get() and mel_spectrogram_for_display is not None:
                # Schedule this on the main thread too
                self.after(0, lambda mel=mel_spectrogram_for_display, cmd=predicted_command: self.display_mel_spectrogram(mel, cmd))
            
            print("DEBUG_THREAD: _record_and_predict_thread successfully finished processing this command.")

        except Exception as e:
            error_msg = f"Recording/Prediction Error in Thread: {e}"
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR_THREAD: {error_msg}") # Console log
            import traceback
            traceback.print_exc() # Print full traceback to console
            self.after(0, lambda msg=error_msg: self.update_status(msg)) # Update status with error
            self.after(0, lambda msg=f"{error_msg}\n": self.update_log_area(msg)) # Log error
            self.after(0, lambda ttl="Thread Error", emsg=error_msg: messagebox.showerror(ttl, emsg)) # Show GUI error box
        finally:
            # Re-enable button from the main thread, regardless of success or failure within try
            self.after(0, lambda: self.record_button.configure(state="normal"))
            print("DEBUG_THREAD: Record button re-enabled in finally block.")

    def display_mel_spectrogram(self, mel_db_norm, title_extra=""):
        print("DEBUG_APP: display_mel_spectrogram called")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db_norm, sr=self.config["sample_rate"],
                                 hop_length=self.config["hop_length"],
                                 x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram (Normalized) - {title_extra}')
        plt.tight_layout()
        plt.show()
        print("DEBUG_APP: display_mel_spectrogram finished, plt.show() called.")
print("DEBUG: After CommandApp class definition")


if __name__ == '__main__':
    print("DEBUG: Script execution started in __main__ block.")
    try:
        print("DEBUG: Setting appearance mode and theme globally (once).")
        ctk.set_appearance_mode("System") 
        ctk.set_default_color_theme("blue")

        import argparse
        print("DEBUG: argparse imported in __main__.")

        parser = argparse.ArgumentParser(description="Voice Command Recognizer with GUI")
        args = parser.parse_args()
        print(f"DEBUG: Parsed args: {args}")

        print("=== Audio Command Recognizer ===") 

        # Use the global CONFIG defined at the top of the script
        dataset_dir_to_check = CONFIG.get("dataset_dir", 'wav_commands') 
        print(f"DEBUG: Checking dataset directory: {dataset_dir_to_check}")
        if not os.path.exists(dataset_dir_to_check):
            print(f"Warning: Dataset directory '{dataset_dir_to_check}' not found.")
            print(f"Please create this directory and add your .wav command files (e.g., 'up_user1_01.wav').")
            # Example: os.makedirs(dataset_dir_to_check, exist_ok=True)
            # print(f"DEBUG: Created directory '{dataset_dir_to_check}'.")

        print("DEBUG: About to create CommandApp instance.")
        app = CommandApp(CONFIG) # Pass the global CONFIG dictionary
        print("DEBUG: CommandApp instance created.")

        print("DEBUG: About to call app.mainloop().")
        app.mainloop()
        print("DEBUG: app.mainloop() finished (should only see this after GUI closes).")

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"AN UNCAUGHT ERROR OCCURRED IN __main__:")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        print("--- TRACEBACK ---")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        input("Press Enter to close this window...")
