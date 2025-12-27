import pandas as pd
import os
import glob
from typing import List

# --- CONFIGURATION (UPDATED) ---
BASE_DIR = '/home/ubuntu/lung_project/TB/' 
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
SOLICITED_AUDIO_DIR = os.path.join(BASE_DIR, 'raw_data', 'solicited_data')


def load_metadata_files(metadata_dir: str) -> pd.DataFrame:
    print("Loading metadata files...")
    
    solicited_meta_path = os.path.join(metadata_dir, 'CODA_TB_Solicited_Meta_Info.csv')
    df_solicited = pd.read_csv(solicited_meta_path)
    
    df_solicited['recording_type'] = 'solicited' 
    
    clinical_meta_path = os.path.join(metadata_dir, 'CODA_TB_Clinical_Meta_Info.csv')
    df_clinical = pd.read_csv(clinical_meta_path)
    
    df_merged = pd.merge(
        df_solicited, 
        df_clinical, 
        on='participant', 
        how='left'
    )
    
    print(f"Loaded and merged {len(df_merged)} cough records with clinical data.")
    return df_merged

def find_audio_files(audio_dir: str) -> List[str]:
    print(f"Searching for .wav files in: {audio_dir}...")
    file_paths = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
    return file_paths

def create_master_dataframe(df_meta: pd.DataFrame, file_paths: List[str]) -> pd.DataFrame:
    print("Creating master DataFrame by combining metadata and file paths...")
    
    df_paths = pd.DataFrame({'audio_path': file_paths})
    df_paths['filename'] = df_paths['audio_path'].apply(os.path.basename)
    
    df_master = pd.merge(
        df_meta, 
        df_paths, 
        on='filename', 
        how='inner' 
    )
    
    return df_master

if __name__ == "__main__":
    
    df_meta_merged = load_metadata_files(METADATA_DIR)
    
    # Using only solicited cough audio paths
    all_audio_paths = find_audio_files(SOLICITED_AUDIO_DIR)
    
    print(f"Found {len(all_audio_paths)} total .wav files in the selected directories.")

    df_master_data = create_master_dataframe(df_meta_merged, all_audio_paths)
    
    # --- SAVE THE MASTER DATAFRAME FOR LATER USE ---
    # File name updated to reflect annotated data paths
    output_csv_path = os.path.join(BASE_DIR, 'annotated_data_paths.csv')
    df_master_data.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully saved the master DataFrame to: {output_csv_path}")
    
    # --- SUMMARY ---
    print("\n--- MASTER DATAFRAME CREATED ---")
    print(f"Total labeled and matched audio files ready for processing: {len(df_master_data)}")
    print(f"TB Positive Samples (tb_status=1): {len(df_master_data[df_master_data['tb_status'] == 1])}")
    print(f"TB Negative Samples (tb_status=0): {len(df_master_data[df_master_data['tb_status'] == 0])}")
    
    print("\nFirst 5 rows of the master DataFrame:")
    print(df_master_data[['participant', 'filename', 'tb_status', 'audio_path']].head())
    
    print("\nNext step: Use the 'audio_path' column to load the raw audio, extract features, and train your model.")
