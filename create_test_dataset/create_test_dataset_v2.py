import os
import pandas as pd
import numpy as np
# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import h5py


print("start")
folder_name = 'ours_all'

# Define the path for the labels CSV file and the expected shape
labels_file_path = 'mfcc_labels.csv'
expected_shape = (40, 32)


# Function to check the shape of each MFCC file
def check_mfcc_shape(file_path):
    try:
        if isinstance(file_path, str) and os.path.exists(file_path):
            mfcc_data = pd.read_csv(file_path, header=None)
            shape = mfcc_data.shape
            if shape == expected_shape:
                return True, mfcc_data
            else:
                print(f"ERROR: {file_path} has incorrect shape: {shape}. Expected: {expected_shape}")
                return False, None
        else:
            print(f"ERROR: Invalid file path: {file_path}")
            return False, None
    except Exception as e:
        print(f"ERROR: Could not read {file_path}. Exception: {e}")
        return False, None


# Load the labels and file paths from the CSV file
labels_data = pd.read_csv(labels_file_path, header=None, names=['mfccs', 'label_value'])

# Initialize lists to hold valid MFCC data, labels, and paths
valid_mfcc_data = []
valid_labels = []
valid_paths = []

# Iterate through each path in the CSV and check the MFCC shape
for index, row in labels_data.iterrows():
    mfcc_file_path = row['mfccs']
    label = row['label_value']
    is_valid, mfcc_data = check_mfcc_shape(mfcc_file_path)
    if is_valid:
        valid_mfcc_data.append(mfcc_data.values)
        valid_labels.append(label)
        valid_paths.append(mfcc_file_path)  # Track the valid paths

# Convert lists to numpy arrays
valid_mfcc_data = np.array(valid_mfcc_data)
valid_labels = np.array(valid_labels)
valid_paths = np.array(valid_paths, dtype='S')  # Save paths as string type

# Create a TensorFlow dataset from the valid MFCC data and labels
mfcc_dataset = tf.data.Dataset.from_tensor_slices((valid_mfcc_data, valid_labels))


# Function to save datasets using h5py
def save_to_h5(data, labels, paths, h5_file_name):
    with h5py.File(h5_file_name, 'w') as h5f:
        # Save the MFCC data
        mfcc_group = h5f.create_group('mfcc')
        for i, mfcc in enumerate(data):
            mfcc_group.create_dataset(str(i), data=mfcc)

        # Save the labels
        label_group = h5f.create_group('label')
        for i, label in enumerate(labels):
            label_group.create_dataset(str(i), data=label)

        # Save the paths as a new dataset
        h5f.create_dataset('test_all_paths', data=paths)


# Save all data, labels, and paths to the HDF5 file
save_to_h5(valid_mfcc_data, valid_labels, valid_paths, f'{folder_name}_test_dataset.h5')
