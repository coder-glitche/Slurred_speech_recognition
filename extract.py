import os
import shutil
from pathlib import Path

# Paths
RAW_DATA_DIR = '/home/yogesh/.cache/kagglehub/datasets/pranaykoppula/torgo-audio/versions/1'  # change to your actual TORGO dataset path
OUTPUT_DIR = 'processed_dataset'
SLURRED_DIR = os.path.join(OUTPUT_DIR, 'slurred')
NON_SLURRED_DIR = os.path.join(OUTPUT_DIR, 'non_slurred')

# Create target folders
os.makedirs(SLURRED_DIR, exist_ok=True)
os.makedirs(NON_SLURRED_DIR, exist_ok=True)

# Define folder mappings
slurred_folders = ['F_Dys', 'M_Dys']
non_slurred_folders = ['F_Con', 'M_Con']

# Allowed extensions
audio_exts = ['.wav', '.mp3']

# Helper function
def process_dir(src_dir, dest_dir, label):
    for root, _, files in os.walk(src_dir):
        for file in files:
            if any(file.endswith(ext) for ext in audio_exts):
                full_path = os.path.join(root, file)
                # Use a simplified file name
                speaker_id = Path(root).parts[-2]  # e.g., FC01
                session_id = Path(root).parts[-1]  # e.g., S01
                mic_type = 'arrayMic' if 'arrayMic' in file else 'headMic'
                new_name = f"{speaker_id}_{session_id}_{mic_type}.wav"

                # Destination path
                dest_path = os.path.join(dest_dir, new_name)
                shutil.copy2(full_path, dest_path)

                print(f"Copied: {full_path} ➜ {dest_path}")

# Process all folders
for folder in slurred_folders:
    process_dir(os.path.join(RAW_DATA_DIR, folder), SLURRED_DIR, 'slurred')

for folder in non_slurred_folders:
    process_dir(os.path.join(RAW_DATA_DIR, folder), NON_SLURRED_DIR, 'non_slurred')

print("\n✅ Dataset has been successfully processed and reorganized.")
