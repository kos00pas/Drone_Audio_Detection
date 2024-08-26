import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras import layers, models
import h5py

folder_name = 'all_no_pitch_no_wind'

def load_h5_dataset(file_name):
    with h5py.File(file_name, 'r') as f:
        mfcc_data = []
        labels = []
        for key in f['mfcc'].keys():
            mfcc_data.append(f['mfcc'][key][()])
            labels.append(f['label'][key][()])

        mfcc_data = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices((mfcc_data, labels))
    return dataset


def prepare_datasets(train_file, val_file, test_file, batch_size=64, shuffle_buffer_size=1000):
    train_dataset = load_h5_dataset(train_file)
    val_dataset = load_h5_dataset(val_file)
    test_dataset = load_h5_dataset(test_file)

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_file = f'./{folder_name}/train_dataset.h5'
    val_file = f'./{folder_name}/val_dataset.h5'
    test_file = f'./{folder_name}/test_dataset.h5'

    print("start")
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_file, val_file, test_file)
    print("done prepare_datasets")

    # Define a simple CNN model for demonstration

    model = models.Sequential([
        layers.Input(shape=(40, 32, 1)),  # Adding a channel dimension for CNN

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
    """Input Shape: Ensure your input data shape matches (40, 32, 1). It is suitable for images with one channel (grayscale).
    Layers and Filters: The chosen layers and filters (Conv2D) are appropriate for extracting features.
    Dense Layers: The Dense layers after flattening are suitable for classification tasks.
    Activation Function: Using 'sigmoid' for the final Dense layer is correct for binary classification.
    Loss Function: 'binary_crossentropy' is the right choice for binary classification.
    Optimizer and Metrics: 'adam' optimizer and 'accuracy' metric are good choices."""
    # model.load_weights('model_.weights.h5')
    print("Model weights loaded.")

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    print("lets fit")
    # Train the model
    import time

    start_time = time.time()

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset
    )
    end_time = time.time()

    print("lets loss")
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc}")

    from datetime import datetime

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the model name dynamically using the timestamp
    model_name = f'./{folder_name}/trained_model_{folder_name}.keras'

    # Save the model in Keras format
    model.save(model_name)
    print(f"Model saved to {model_name}")
    # Save the training history using the same timestamp
    history_name = f'./{folder_name}/history_{folder_name}.pkl'
    with open(history_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print(f"Training history saved to {history_name}")

    # Load the model
    loaded_model = models.load_model(model_name)
    print(f"Model loaded from {model_name}")

    model.save_weights(f'./{folder_name}/model12_{folder_name}.weights.h5')
    print("Model weights saved.")

    # Evaluate the loaded model
    test_loss, test_acc = loaded_model.evaluate(test_dataset)
    print(f"Test Accuracy of the loaded model: {test_acc}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")