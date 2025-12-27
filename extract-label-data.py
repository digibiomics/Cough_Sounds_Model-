import os
import pandas as pd
import librosa
from tqdm import tqdm
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = '/home/ubuntu/lung_project/TB/annotated_data_paths.csv'    # Edit as required
AUDIO_PATH_COLUMN = 'audio_path'
LABEL_COLUMN = 'tb_status'

# --- LOAD CSV ---
df = pd.read_csv(CSV_PATH)

# --- AUDIO ANALYSIS ---
durations = []
sample_rates = []

print("Analyzing audio files for sample rate and duration...")
for audio_fp in tqdm(df[AUDIO_PATH_COLUMN].values):
    try:
        y, sr = librosa.load(audio_fp, sr=None, mono=True)
        durations.append(librosa.get_duration(y=y, sr=sr))
        sample_rates.append(sr)
    except Exception as e:
        durations.append(np.nan)
        sample_rates.append(None)
        print(f"Failed to process: {audio_fp}, error: {e}")

df['duration_sec'] = durations
df['sample_rate'] = sample_rates

# --- DATASET SUMMARY ---
print(f"\n--- LABEL SUMMARY ---")
print(f"Total samples: {len(df)}")
print(f"Participants: {df['participant'].nunique()}")
print(df[LABEL_COLUMN].value_counts())

print(f"\n--- SAMPLE RATE SUMMARY ---")
print(df['sample_rate'].value_counts())

print(f"\n--- DURATION STATISTICS (in seconds) ---")
print(df['duration_sec'].describe())

print(f"\nDuration quartiles:\n{df['duration_sec'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])}")

# --- HISTOGRAM ---
import matplotlib.pyplot as plt

plt.hist(df['duration_sec'].dropna(), bins=50)
plt.xlabel("Duration (seconds)")
plt.ylabel("File Count")
plt.title("Distribution of Audio File Durations")
plt.show()

# Save enhanced DataFrame for later processing
df.to_csv('/path/to/your/annotated_data_with_audio_stats.csv', index=False)
