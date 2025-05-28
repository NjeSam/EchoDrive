import os
import librosa
import numpy as np
import soundfile as sf # For saving audio, more robust than scipy.io.wavfile for float32
import shutil

# Configuration for augmentation
SOURCE_DIR = 'wav_commands'
TARGET_DIR = 'wav_commands_augmented' # Save to a new dir first, then merge if desired
SAMPLE_RATE = 16000 # Must match your model's sample rate

# Augmentation parameters
AUGMENTATIONS_PER_FILE = 3 # How many new versions to create per original file
NOISE_LEVELS = [0.001, 0.003, 0.005] # Relative to max amplitude
PITCH_SHIFTS = [-2, -1, 1, 2] # Semitones
TIME_STRETCH_RATES = [0.85, 0.9, 1.1, 1.15]

def add_noise(y, sr, level_factor):
    """Adds random Gaussian noise."""
    if np.amax(y) == 0: return y # Avoid division by zero for silent audio
    noise_amp = level_factor * np.random.uniform(0.5, 1.0) * np.amax(y)
    noise = noise_amp * np.random.normal(size=len(y))
    return y + noise

def change_pitch(y, sr, n_steps):
    """Changes pitch."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def stretch_time(y, sr, rate):
    """Stretches time."""
    return librosa.effects.time_stretch(y, rate=rate)

def main_augmenter():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    if os.path.exists(TARGET_DIR):
        print(f"Warning: Target directory '{TARGET_DIR}' already exists. Content might be overwritten or mixed.")
    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Starting augmentation from '{SOURCE_DIR}' to '{TARGET_DIR}'...")
    file_count = 0
    augmented_count = 0

    for filename in os.listdir(SOURCE_DIR):
        if not filename.lower().endswith('.wav'):
            continue

        source_filepath = os.path.join(SOURCE_DIR, filename)
        base, ext = os.path.splitext(filename)

        try:
            y, sr = librosa.load(source_filepath, sr=SAMPLE_RATE)
            file_count += 1

            # --- Create augmented versions ---
            # You can choose to apply one type of augmentation per file, or multiple sequentially.
            # For simplicity, let's create distinct augmented files for each type.

            # 1. Noise augmentation
            for i, level in enumerate(NOISE_LEVELS):
                if augmented_count % AUGMENTATIONS_PER_FILE == 0 and i > 0: # Limit total number
                    break
                y_noise = add_noise(y.copy(), sr, level)
                target_filename_noise = f"{base}_noise{i+1}{ext}"
                target_filepath_noise = os.path.join(TARGET_DIR, target_filename_noise)
                sf.write(target_filepath_noise, y_noise, sr)
                augmented_count += 1
                # print(f"  Generated: {target_filename_noise}")

            # 2. Pitch shift augmentation
            for i, n_steps in enumerate(PITCH_SHIFTS):
                if augmented_count % AUGMENTATIONS_PER_FILE == 0 and i > 0:
                    break
                y_pitch = change_pitch(y.copy(), sr, n_steps)
                target_filename_pitch = f"{base}_pitch{i+1}{ext}"
                target_filepath_pitch = os.path.join(TARGET_DIR, target_filename_pitch)
                sf.write(target_filepath_pitch, y_pitch, sr)
                augmented_count +=1
                # print(f"  Generated: {target_filename_pitch}")

            # 3. Time stretch augmentation
            for i, rate in enumerate(TIME_STRETCH_RATES):
                if augmented_count % AUGMENTATIONS_PER_FILE == 0 and i > 0:
                    break
                y_stretch = stretch_time(y.copy(), sr, rate)
                target_filename_stretch = f"{base}_stretch{i+1}{ext}"
                target_filepath_stretch = os.path.join(TARGET_DIR, target_filename_stretch)
                sf.write(target_filepath_stretch, y_stretch, sr)
                augmented_count += 1
                # print(f"  Generated: {target_filename_stretch}")
            
            # Optionally, copy original file too
            # shutil.copy2(source_filepath, os.path.join(TARGET_DIR, filename))


        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nAugmentation complete.")
    print(f"Processed {file_count} original files.")
    print(f"Generated approximately {augmented_count} augmented files in '{TARGET_DIR}'.")
    print(f"Consider reviewing files in '{TARGET_DIR}' and then manually merging them into '{SOURCE_DIR}' or updating your CONFIG['dataset_dir'].")

if __name__ == '__main__':
    # Before running, ensure you have soundfile: pip install soundfile
    main_augmenter()
