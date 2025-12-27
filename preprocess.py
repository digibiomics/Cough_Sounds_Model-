import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# CONFIGURATION
CSV_PATH = '/home/ubuntu/lung_project/TB/annotated_data_paths.csv'
AUDIO_COLUMN = 'audio_path'
LABEL_COLUMN = 'tb_status'
SR = 44100
DURATION = 0.5
N_MFCC = 40

# Load data
df = pd.read_csv(CSV_PATH)

# Balance classes by undersampling negatives
df_pos = df[df[LABEL_COLUMN] == 1]
df_neg = df[df[LABEL_COLUMN] == 0]
df_neg_down = resample(df_neg, n_samples=len(df_pos), replace=False, random_state=42)
df_balanced = pd.concat([df_pos, df_neg_down]).sample(frac=1, random_state=42).reset_index(drop=True)

# Feature extraction function
def extract_mfcc(audio_fp):
    y, sr = librosa.load(audio_fp, sr=SR, mono=True)
    y = librosa.util.normalize(y)
    max_len = int(SR * DURATION)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.flatten()

# Extract features
print("Extracting MFCC features...")
X = []
y = []
for idx, row in df_balanced.iterrows():
    mfcc = extract_mfcc(row[AUDIO_COLUMN])
    X.append(mfcc)
    y.append(row[LABEL_COLUMN])
X = np.array(X)
y = np.array(y)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
print("Preprocessed data saved: X_train.npy, y_train.npy, X_val.npy, y_val.npy")
