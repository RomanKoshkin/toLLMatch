import os
import sys
from pydub import AudioSegment

def get_audio_length(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio.duration_seconds

exp_dir = sys.argv[1]
directory = exp_dir + "/wavs"

audio_lengths = []

for file_name in os.listdir(directory):
    if file_name.endswith('.wav'):
        file_path = os.path.join(directory, file_name)
        audio_length = get_audio_length(file_path)
        audio_lengths.append((audio_length, file_name))

# 長さ順に並べ替え
sorted_audio_lengths = sorted(audio_lengths, key=lambda x: x[0])

# 並べ替えたリストを表示
for audio_length, file_name in sorted_audio_lengths:
        print(f"{file_name}: {audio_length:.2f} seconds")
