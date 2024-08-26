import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import h5py


print("start")


# Load the HDF5 dataset
def load_h5_dataset(file_name):
    with h5py.File(file_name, 'r') as f:
        mfcc_data = []
        labels = []
        for key in f['mfcc'].keys():
            mfcc_data.append(f['mfcc'][key][()])
            labels.append(f['label'][key][()])

        mfcc_data = np.array(mfcc_data)
        labels = np.array(labels)

    return mfcc_data, labels


# Load the dataset
train_file = '../v8/no_ours_no_abc/train_dataset.h5'
val_file = '../v8/no_ours_no_abc/val_dataset.h5'

X_train, y_train = load_h5_dataset(train_file)
X_val, y_val = load_h5_dataset(val_file)

# Ensure that the data has the correct shape
X_train = X_train.reshape(-1, 40, 32, 1)
X_val = X_val.reshape(-1, 40, 32, 1)


# Function to build the model
def build_model(optimizer='adam'):
    model = keras.Sequential([
        layers.Input(shape=(40, 32, 1)),  # Input layer with the correct shape

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define the hyperparameters grid to search
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'epochs': [10, 20]
}

# Manual Grid Search
best_accuracy = 0
best_params = {}
best_model = None

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

for optimizer in param_grid['optimizer']:
    for epochs in param_grid['epochs']:
        print(f"Training with: optimizer={optimizer}, epochs={epochs}")

        # Build and train the model with the current set of hyperparameters
        model = build_model(optimizer=optimizer)
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1,
                            callbacks=[early_stopping])

        # Evaluate on validation data
        val_accuracy = max(history.history['val_accuracy'])

        # Check if this is the best model so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {
                'optimizer': optimizer,
                'epochs': epochs
            }
            best_model = model  # Save the best model

# Save the best model
if best_model is not None:
    best_model.save('best_cnn_model.h5')
    print(f"Best model saved as 'best_cnn_model.h5'")

# Print the best result
print(f"Best validation accuracy: {best_accuracy} with parameters: {best_params}")
