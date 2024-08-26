import pandas as pd
import librosa
import numpy as np
import os

def augment_audio(y, sr):
    y = y.astype(np.float32)  # Ensure the audio is in floating-point format
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    y_stretch = y_stretch[:len(y)]  # Ensure the stretched signal has the same length
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)[:len(y)]  # Ensure the pitch-shifted signal has the same length
    return [y, y_noise, y_stretch, y_shift]

def augment_and_save(input_dir, sr=16000):
    for root, dirs, files in os.walk(input_dir):
        if 'signal.csv' in files and root.endswith('_noise'):
            signal_path = os.path.join(root, 'signal.csv')
            print(signal_path)
            signal_df = pd.read_csv(signal_path, header=None)
            signal = signal_df.values.flatten().astype(np.float32)  # Convert to float32

            augmented_signals = augment_audio(signal, sr)
            suffixes = ["", "_noised", "_time-stretched", "_pitch-shifted"]

            for augmented_signal, suffix in zip(augmented_signals, suffixes):
                augmented_signal_df = pd.DataFrame(augmented_signal)

                # Fill NaN values with 0
                augmented_signal_df.fillna(0, inplace=True)

                # Ensure the dtype is the same as the original
                augmented_signal_df = augmented_signal_df.astype(signal_df.dtypes[0])

                output_path = os.path.join(root, f'signal{suffix}.csv')
                augmented_signal_df.to_csv(output_path, header=False, index=False)
                print(f"Saved augmented signal to: {output_path}")

def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        import json
        data = json.load(file)
        return data["folder_name"]


root_directory =get_folder_name()
cleaned_root_directory = root_directory.replace('../', '')
last_dir= '../FINISHED_V7/'+cleaned_root_directory

# Run the augmentation and saving process
augment_and_save(last_dir)
