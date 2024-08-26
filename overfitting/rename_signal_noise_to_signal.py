import os
def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        import json
        data = json.load(file)
        return data["folder_name"]


root_directory =get_folder_name()
cleaned_root_directory = root_directory.replace('../', '')
root_dir= '../FINISHED_V7/'+cleaned_root_directory


# Walk through all_without_eac directories and files in the root directory
for subdir, _, files in os.walk(root_dir):
    # Check if the subdirectory name ends with '_noise'
    if subdir.endswith('_noise'):
        for file in files:
            # Check if the file name is 'signal_noise.csv'
            if file == 'signal_noise.csv':
                old_file_path = os.path.join(subdir, file)
                new_file_path = os.path.join(subdir, 'signal.csv')
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} to {new_file_path}')
