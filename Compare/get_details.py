import os
# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import os


import csv
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc,
                             precision_recall_curve, cohen_kappa_score)
import pickle

print("start")
models_folder_names = [
                'all_no_pitch_no_wind'
]
test_model_folder_path='../create_test_dataset/ours_all_test_dataset.h5'

for folder_name in models_folder_names:

    output_dir = f'../Create_Dataset_and_train/{folder_name}/outputs'
    #Create_Dataset_and_train / all_wind_with_extra / trained_model_all_wind_with_extra.keras
    misclassified_dir = os.path.join(output_dir, 'misclassified_samples')

    # Create output directory and misclassified samples directory if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(misclassified_dir, exist_ok=True)

    # Step 1: Load the trained model
    model = tf.keras.models.load_model(f'../Create_Dataset_and_train/{folder_name}/trained_model_{folder_name}.keras')

    # Step 2: Load the test dataset from the .h5 file
    with h5py.File(test_model_folder_path, 'r') as h5f:
        mfcc_group = h5f['mfcc']
        label_group = h5f['label']
        paths_group = h5f['test_all_paths']  # New dataset containing the paths

        test_data = np.array([mfcc_group[str(i)][:] for i in range(len(mfcc_group))])
        test_labels = np.array([label_group[str(i)][()] for i in range(len(label_group))])

        # Function to get the path of a sample given its index
        # Function to get the path of a sample given its index
        def get_sample_path(index):
            return paths_group[index].decode('utf-8')  # Decode the byte string to get the path


        # Step 3: Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(test_data, test_labels)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

        # Step 4: Generate predictions and calculate metrics
        predictions = model.predict(test_data)
        threshold = 0.5  # You can adjust this threshold as needed
        predicted_classes = (predictions > threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(test_labels, predicted_classes)

        # Calculate False Positives (FP) and False Negatives (FN)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)

        # Save FP and FN to CSV
        fp_fn_csv_file_path = os.path.join(output_dir, 'fp_fn_report.csv')
        with open(fp_fn_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            header = ['class', 'false_positives', 'false_negatives']
            writer.writerow(header)

            # Write FP and FN for each class
            for i in range(len(FP)):
                writer.writerow([f'Class {i}', FP[i], FN[i]])

        print(f"FP and FN results saved to {fp_fn_csv_file_path}")

        # Classification Report (Printed to console and saved to CSV)
        report = classification_report(test_labels, predicted_classes, output_dict=True,
                                       target_names=[f'Class {i}' for i in np.unique(test_labels)], zero_division=0)
        print("Classification Report:\n", classification_report(test_labels, predicted_classes))

        # Save all relevant data to a CSV file
        csv_file_path = os.path.join(output_dir, 'classification_report.csv')

        # Write or append to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header only if file is empty
            if file.tell() == 0:
                header = ['model_name', 'test_loss', 'test_accuracy']
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric_name in metrics.keys():
                            header.append(f"{label}_{metric_name}")
                    else:
                        header.append(label)
                writer.writerow(header)

            # Write the results
            row = [folder_name, loss, accuracy]
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    row.extend([metrics[metric_name] for metric_name in metrics.keys()])
                else:
                    row.append(metrics)
            writer.writerow(row)

        print(f"Results saved to {csv_file_path}")

        # Identify misclassified samples and save their paths with correct labels
        false_positive_paths = []
        false_negative_paths = []

        for idx, (true_label, pred_class) in enumerate(zip(test_labels, predicted_classes)):
            # Get the path of the specific sample using the index
            path = get_sample_path(idx)
            if true_label == 0 and pred_class == 1:  # False Positive
                false_positive_paths.append((path, true_label))
            elif true_label == 1 and pred_class == 0:  # False Negative
                false_negative_paths.append((path, true_label))

        # Write false positives and false negatives to separate text files
        with open(os.path.join(misclassified_dir, 'false_positives.txt'), 'w') as fp_file:
            for path, label in false_positive_paths:
                fp_file.write(f"{path}, Correct Label: {label}\n")

        with open(os.path.join(misclassified_dir, 'false_negatives.txt'), 'w') as fn_file:
            for path, label in false_negative_paths:
                fn_file.write(f"{path}, Correct Label: {label}\n")

    print(f"Misclassified sample paths saved in {misclassified_dir}")

    # Optional: Plotting (as in the original code)
    # Creating subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    # Add a super title to the figure with the folder_name
    fig.suptitle(f' {folder_name}', fontsize=16, weight='bold', y=1.02)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=axs[0, 0])
    axs[0, 0].set_title('Confusion Matrix')

    # Load and plot training history
    with open(f'../Create_Dataset_and_train/{folder_name}/history_{folder_name}.pkl', 'rb') as file:
        history = pickle.load(file)

    # Plot Training and Validation Loss/Accuracy
    train_loss = history['loss'][-1]
    val_loss = history['val_loss'][-1]
    train_acc = history['accuracy'][-1]
    val_acc = history['val_accuracy'][-1]

    axs[0, 1].text(0.1, 0.8, f"Train Loss: {train_loss:.4f}", fontsize=12, weight='bold')
    axs[0, 1].text(0.1, 0.6, f"Validation Loss: {val_loss:.4f}", fontsize=12, weight='bold')
    axs[0, 1].text(0.1, 0.4, f"Train Accuracy: {train_acc:.4f}", fontsize=12, weight='bold')
    axs[0, 1].text(0.1, 0.2, f"Validation Accuracy: {val_acc:.4f}", fontsize=12, weight='bold')
    axs[0, 1].text(0.1, 0.0, f"Test Loss: {loss:.4f}", fontsize=12, weight='bold')
    axs[0, 1].text(0.1, -0.2, f"Test Accuracy: {accuracy:.4f}", fontsize=12, weight='bold')
    axs[0, 1].axis('off')

    # ROC Curve (for binary classification)
    if len(np.unique(test_labels)) == 2:  # Only plot ROC if binary classification
        if predictions.shape[1] == 1:  # Handle single probability output
            fpr, tpr, _ = roc_curve(test_labels, predictions)
        else:
            fpr, tpr, _ = roc_curve(test_labels, predictions[:, 1])

        roc_auc = auc(fpr, tpr)
        axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[1, 0].set_xlabel('False Positive Rate')
        axs[1, 0].set_ylabel('True Positive Rate')
        axs[1, 0].set_title('Receiver Operating Characteristic')
        axs[1, 0].legend(loc="lower right")

    # Precision-Recall Curve
    if len(np.unique(test_labels)) == 2:  # Binary classification
        if predictions.shape[1] == 1:  # Handle single probability output
            precision, recall, _ = precision_recall_curve(test_labels, predictions)
        else:
            precision, recall, _ = precision_recall_curve(test_labels, predictions[:, 1])

        axs[1, 1].plot(recall, precision, marker='.')
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].set_title('Precision-Recall Curve')

    # Plotting the loss curves
    axs[2, 0].plot(history['loss'], label='Training Loss')
    axs[2, 0].plot(history['val_loss'], label='Validation Loss')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Loss')
    axs[2, 0].set_title('Loss Curves')
    axs[2, 0].legend()

    # Plotting the accuracy curves
    axs[2, 1].plot(history['accuracy'], label='Training Accuracy')
    axs[2, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].set_ylabel('Accuracy')
    axs[2, 1].set_title('Learning Curves (Accuracy)')
    axs[2, 1].legend()

    # Finalize and save the entire figure with all subplots
    full_plot_path = os.path.join(output_dir, 'full_evaluation_plots.png')
    plt.tight_layout()
    plt.savefig(full_plot_path)
    plt.show()
