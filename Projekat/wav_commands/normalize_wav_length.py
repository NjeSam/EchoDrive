from pydub import AudioSegment
import os

folder = "."
target_duration = 3000  # dolžina v milisekundah (3 sekunde)

for file in os.listdir(folder):
    if file.endswith(".wav"):
        path = os.path.join(folder, file)
        audio = AudioSegment.from_wav(path)

        if len(audio) < target_duration:
            silence = AudioSegment.silent(duration=target_duration - len(audio))
            padded = audio + silence
            padded.export(path, format="wav")
            print(f"{file} -> podaljšan na 3s")
        else:
            trimmed = audio[:target_duration]
            trimmed.export(path, format="wav")
            print(f"{file} -> odrezan na 3s")

