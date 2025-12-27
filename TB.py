import kagglehub
import os

# --- Configuration ---
KAGGLE_DATASET_HANDLE = "ruchikashirsath/tb-audio"
# The dataset will be downloaded to the KaggleHub cache directory.
# 'dataset_path' will store the exact local path to the downloaded files.

# --- Download the full dataset folder ---
# This downloads the dataset to your local cache and returns the path.
print(f"Starting download of dataset: {KAGGLE_DATASET_HANDLE}")
local_dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)

print(f"\n Download Complete!")
print(f"Local Path to Dataset Files: {local_dataset_path}")

# The contents of the 'tb-audio' dataset include a folder named 'TB-Audio'.
# The full local path to the audio files will be:
# local_directory_to_upload = os.path.join(local_dataset_path, "TB-Audio")
# print(f"Full directory path for S3 upload: {local_directory_to_upload}")

# Note: The 'TB-Audio' directory is what you will upload to S3.