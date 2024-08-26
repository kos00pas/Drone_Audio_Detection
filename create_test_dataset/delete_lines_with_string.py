import pandas as pd
print("start")



# Load the CSV file
file_path = 'mfcc_labels.csv'
df = pd.read_csv(file_path)

"""String to delete , if is folder  use the the r'\\<folder_name>' """
# string_to_delete = 'signal_pitch-shifted.csv'
string_to_delete = r'\\ours_3'

# df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains(string_to_delete).any(), axis=1)]
df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains(string_to_delete).any(), axis=1)]

df_filtered.to_csv(file_path, index=False)

print(f"Lines with {string_to_delete} have been removed.")
