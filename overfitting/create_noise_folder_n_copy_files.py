import os
import shutil
import json
def create_noise_folders_and_copy_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            new_dir = os.path.join(dirpath, dirname + "_noise")
            os.makedirs(new_dir, exist_ok=True)
            print(f"Created: {new_dir}")

        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            # Ensure the destination directory exists
            destination_dir = dirpath + "_noise"
            os.makedirs(destination_dir, exist_ok=True)
            destination_file = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")

def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]


root_directory =get_folder_name()
cleaned_root_directory = root_directory.replace('../', '')
last_dir= '../FINISHED_V7/'+cleaned_root_directory

# print(root_directory,cleaned_root_directory,last_dir) ; exit()

create_noise_folders_and_copy_files(last_dir)
