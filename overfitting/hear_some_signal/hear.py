import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import os

# List of file paths
file_paths = [
    '../FINISHED_V7/wind_data/Windnoise2 - 99 - Audio -  19.8 0 -10dB_resampled_chunk_32_noise/signal.csv',
    '../FINISHED_V7/wind_data/Windnoise2 - 99 - Audio -  19.8 0 -10dB_resampled_chunk_32_noise/signal_noised.csv',
    '../FINISHED_V7/wind_data/Windnoise2 - 99 - Audio -  19.8 0 -10dB_resampled_chunk_32_noise/signal_time-stretched.csv',
    '../FINISHED_V7/wind_data/Windnoise2 - 99 - Audio -  19.8 0 -10dB_resampled_chunk_32/signal.csv'
]

# Directory to save WAV files
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Processing each file
for file_path in file_paths:
    # Step 1: Read the CSV file
    try:
        data = pd.read_csv(file_path, dtype=np.int16)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue

    # Step 2: Convert DataFrame to NumPy array
    audio_data = data.values.flatten()

    # Step 3: Create a corresponding output file name
    base_name = file_path.replace('.csv', '.wav').replace('../', '').replace('/', '_')
    output_wav = os.path.join(output_dir, base_name)

    # Step 4: Write the data to a WAV file
    sample_rate = 16000  # Adjust based on your data's original sampling rate
    write(output_wav, sample_rate, audio_data)

    print(f"WAV file saved as {output_wav}")
