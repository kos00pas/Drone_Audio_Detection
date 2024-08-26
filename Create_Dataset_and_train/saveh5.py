import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
folder_name = 'all_no_pitch_no_wind'

print("Loading CSV files...")

# Load the labels and file paths from mfccs_labels.csv
labels_file_path = f'./{folder_name}/mfcc_labels_{folder_name}.csv'
labels_data = pd.read_csv(labels_file_path, header=None, names=['mfccs', 'label_value'])
print("i")

# Define the expected shape
expected_shape = (40, 32)

# Function to check the shape of each MFCC file
def check_mfcc_shape(file_path):
    try:
        if isinstance(file_path, str) and os.path.exists(file_path):
            mfcc_data = pd.read_csv(file_path, header=None)
            shape = mfcc_data.shape
            if shape == expected_shape:
                print("ok")
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
print("i")

# Initialize lists to hold valid MFCC data_mp3 and labels
valid_mfcc_data = []
valid_labels = []

# Iterate through each path in mfccs_labels.csv and check the MFCC shape
for index, row in labels_data.iterrows():
    mfcc_file_path = row['mfccs']
    label = row['label_value']
    is_valid, mfcc_data = check_mfcc_shape(mfcc_file_path)
    if is_valid:
        valid_mfcc_data.append(mfcc_data.values)
        valid_labels.append(label)
print("i")

# Convert lists to numpy arrays
valid_mfcc_data = np.array(valid_mfcc_data)
valid_labels = np.array(valid_labels)

# Print the total number of valid samples
print(f"Total valid samples: {len(valid_mfcc_data)}")

# Create a TensorFlow dataset from the valid MFCC data_mp3 and labels
mfcc_dataset = tf.data.Dataset.from_tensor_slices(valid_mfcc_data)
labels_dataset = tf.data.Dataset.from_tensor_slices(valid_labels)
dataset = tf.data.Dataset.zip((mfcc_dataset, labels_dataset))
print("i")

# Function to count elements in a dataset
@tf.autograph.experimental.do_not_convert
def count_elements(dataset):
    return dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
print("i")

# Calculate the sizes of each split
filtered_dataset_size = count_elements(dataset)
train_size = int(0.7 * filtered_dataset_size)
val_size = int(0.15 * filtered_dataset_size)
test_size = filtered_dataset_size - train_size - val_size
print("i")
# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=filtered_dataset_size, reshuffle_each_iteration=False)
print("i")

# Split the dataset
train_dataset = dataset.take(train_size)
val_test_dataset = dataset.skip(train_size)
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size)

# Ensure there are elements in the datasets
print(f"Train dataset size: {count_elements(train_dataset)}")
print(f"Validation dataset size: {count_elements(val_dataset)}")
print(f"Test dataset size: {count_elements(test_dataset)}")

# Function to save datasets using h5py
def save_to_h5(dataset, file_name):
    with h5py.File(file_name, 'w') as f:
        mfcc_grp = f.create_group('mfcc')
        label_grp = f.create_group('label')
        for i, (mfcc, label) in enumerate(dataset):
            mfcc_np = mfcc.numpy()
            label_np = label.numpy()
            mfcc_grp.create_dataset(str(i), data=mfcc_np)
            label_grp.create_dataset(str(i), data=label_np)
            # if i < 5:  # Print first few samples for debugging
            #     print(f"Saved sample {i}: MFCC shape {mfcc_np.shape}, label {label_np}")
print("i")

# Save the datasets to HDF5 files
save_to_h5(train_dataset, f'{folder_name}/train_dataset.h5')
save_to_h5(val_dataset, f'{folder_name}/val_dataset.h5')
save_to_h5(test_dataset, f'{folder_name}/test_dataset.h5')

print("Datasets saved to HDF5")

# Function to count labels in an HDF5 file
def count_labels_in_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        labels = [f['label'][key][()] for key in f['label'].keys()]
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        return label_counts

# Count labels in each HDF5 file
train_label_counts = count_labels_in_h5(f'{folder_name}/train_dataset.h5')
val_label_counts = count_labels_in_h5(f'{folder_name}/val_dataset.h5')
test_label_counts = count_labels_in_h5(f'{folder_name}/test_dataset.h5')

# Print label counts
print(f"Train dataset label counts: {train_label_counts}")
print(f"Validation dataset label counts: {val_label_counts}")
print(f"Test dataset label counts: {test_label_counts}")