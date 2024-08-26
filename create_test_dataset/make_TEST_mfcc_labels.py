import os
import pandas as pd


def process_directories(base_path, output_file):
    with open(output_file, 'a') as out_file:
        for root, dirs, files in os.walk(base_path):
            label_path = os.path.join(root, 'label.csv')

            if os.path.isfile(label_path):
                label_df = pd.read_csv(label_path, header=None)

                if not label_df.empty and len(label_df.columns) > 0:
                    label = label_df.iloc[0, 0]
                    if label == 'drone':
                        label_value = 1
                    elif label == 'not_drone':
                        label_value = 0
                    else:
                        print(f"Warning: Not eligible label found in {label_path}")
                        label_value = 'not eligible'

                    for file in files:
                        if file.startswith('mfcc'):
                            mfcc_path = os.path.join(root, file)
                            out_file.write(f"{mfcc_path},{label_value}\n")
                else:
                    print(f"Warning: Label column not found or empty in {label_path}")


base_path = '../FINISHED_V7/ours_3'
output_file = 'mfcc_labels.csv'
print("start")
process_directories(base_path, output_file)
