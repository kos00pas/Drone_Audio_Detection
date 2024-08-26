import os
import numpy as np
import pandas as pd
import librosa

def make_mfcc(file_path, save_path, n_mfcc=40, n_fft=2048, hop_length=512, target_frames=32):
    """
    Scope:
        1. Get corresponding data and their details
        2. Normalize data to a reference level
        3. Make the MFCC through librosa library
        4. Pad or truncate MFCC to a fixed length
        5. Save MFCC to a CSV file
    """
    print(f'Making MFCC for {file_path}.....wait')
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Convert the data to a numpy array
    audio_data = data.values.flatten().astype(np.float32)

    # Parameters
    sr = 16000  # Assuming a sample rate of 16000 Hz, update if different

    # Check for NaN or Inf values and replace them with zeros
    audio_data = np.nan_to_num(audio_data)

    # Normalize the entire audio to a reference level (e.g., -20 dB)
    max_val = np.max(np.abs(audio_data))
    if np.isfinite(max_val) and max_val != 0:
        audio_data /= max_val
        audio_data *= 10 ** (-20 / 20)  # Reference level at -20 dB

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmax=8000)

    # Pad or truncate MFCC to ensure consistent shape
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]

    print(f"Done MFCC for {file_path}")

    # Save MFCC to CSV without header
    mfcc_df = pd.DataFrame(mfcc)
    mfcc_df.to_csv(save_path, index=False, header=False)

def process_directories(root_dir, target_frames=32):
    """
    Process each subdirectory in the root directory to find signal.csv
    and create mfcc.csv in the same directory.
    """
    file_names = ['signal_noised.csv', 'signal_pitch-shifted.csv', 'signal_time-stretched.csv']

    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith('_noise'):
            for file in files:
                if file in file_names:
                    file_path = os.path.join(subdir, file)
                    save_path = os.path.join(subdir, f'mfcc_{file}')
                    make_mfcc(file_path, save_path, target_frames=target_frames)

def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        import json
        data = json.load(file)
        return data["folder_name"]


root_directory =get_folder_name()
cleaned_root_directory = root_directory.replace('../', '')
last_dir= '../FINISHED_V7/'+cleaned_root_directory

# Set the root directory
# all_data = '../FINISHED_V7/mistakes'

# Example usage
process_directories(last_dir)
