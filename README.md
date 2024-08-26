
# Classification_Training_Drone_Audio_Detection


- **Requirements**: 
  - pandas, numpy, tensorflow, sklearn, matplotlib, h5py, seaborn, pickle
  - csv, shutil, librosa, scipy, pydub, wave, soundfile 
- **Environment**:
  - conda 24.5.0
  - Python 3.9.19
    - pandas: 2.2.2
    - numpy: 1.26.4
    - tensorflow: 2.17.0
    - scikit-learn (sklearn): 1.5.1
    - matplotlib: 3.9.1
    - h5py: 3.11.0
    - seaborn: 0.13.2
    - librosa: 0.10.2.post1
    - scipy: 1.13.1
    - pydub: 0.25.1
    - soundfile: 0.12.1

To install these dependencies:

```bash
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 matplotlib==3.9.1 seaborn==0.13.2 h5py==3.11.0 tensorflow==2.17.0 scikit-learn==1.5.1 librosa==0.10.2.post1 pydub==0.25.1 soundfile==0.12.1
```

## Pre-Process, Labeling of Data & Overfitting  

### Dataset 
1. **SOUND-BASED DRONE FAULT CLASSIFICATION USING MULTI-TASK LEARNING (Version 1)**  
   29th International Congress on Sound and Vibration (ICSV29), Wonjun Yi, Jung-Woo Choi, & Jae-Woo Lee. (2023). Prague. [Zenodo](https://doi.org/10.5281/zenodo.7779574)
   
2. **DREGON**  
   Audio-Based Search and Rescue with a Drone: Highlights from the IEEE Signal Processing Cup 2019 Student Competition. Antoine Deleforge, Diego Di Carlo, Martin Strauss, Romain Serizel, & Lucio Marcenaro. (2019). IEEE Signal Processing Magazine, 36(5), 138-144. Institute of Electrical and Electronics Engineers. [Kaggle](https://www.kaggle.com/datasets/awsaf49/ieee-signal-processing-cup-2019-dataset)

3. **Audio Based Drone Detection and Identification using Deep Learning**  
   Sara A Al-Emadi, Abdulla K Al-Ali, Abdulaziz Al-Ali, Amr Mohamed. [GitHub](https://github.com/saraalemadi/DroneAudioDataset/tree/master)

4. **DronePrint**  
   Harini Kolamunna, Thilini Dahanayake, Junye Li, Suranga Seneviratne, Kanchana Thilakaratne, Albert Y. Zomaya, Aruna Seneviratne. [GitHub](https://github.com/DronePrint/DronePrint/tree/master)

5. **Drone Detection and Classification using Machine Learning and Sensor Fusion**  
   Svanström F. (2020). [GitHub](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/master)

6. **DroneNoise Database**  
   Carlos Ramos-Romero, Nathan Green, César Asensio and Antonio J Torija Martinez. [Figshare](https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411)

7. **ESC: Dataset for Environmental Sound Classification**  
   Piczak, Karol J. [GitHub](https://github.com/karolpiczak/ESC-50)

8. **drone-audio-detection**  
   [GitHub](https://github.com/BowonY/drone-audio-detection/tree/develop)

9. **Our dataset** -> ours, ours_2, ours_3 

10. **Mistakes** -> faults from training  

## Architecture of CNN 
- Input: MFCC features from audio (shape: (40, 32, 1)).
- Output: Probability of classifying the audio as drone or not drone.

```
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
```

### Hyperparameters
- Running `hyperparameters/find_hypr.py` we find that the best parameters are: 
    - epochs=10
    - optimizer='adam' (default learning rate of 0.001)
    - loss='binary_crossentropy'
    - metrics=['accuracy']

### Preprocess
#### Folder: `Classification_Training_Drone_Audio_Detection/pre-process-code`

When you get a new dataset, pre-process the data with the following order to prepare it for training:

1. **Copy the original data in the root of .git:**
   - `Classification_Training_Drone_Audio_Detection`
   - e.g., folder `Classification_Training_Drone_Audio_Detection/data_mp3`

2. **Change folder_name in file:**
   - `Classification_Training_Drone_Audio_Detection/pre-process-code/file_with_folder_name.json`
   - e.g., `"folder_name": "../data_mp3"`

3. If the data are in mp3, convert them to .wav:
   - `convertion_mp3_wav.py`

4. Convert data to sample rate -> 16kHz:
   - `resampling.py`

5. Split the data in 1sec duration:
   - `splitting.py`

6. Create a directory for each 1 sec to contain:
   1. `signal.csv` (16000,1), dtype=np.int16 based on recording of respeaker dtype:float32
   2. `mfcc.csv` for training model
   - `create_signal_csv.py`
   - `make_mfcc.py`

### Labeling
#### Folder: `Classification_Training_Drone_Audio_Detection/pre-process-code/for_labels`

For each signal.csv, create a label.csv with values: `'drone'` or `'not_drone'`.

1. Take all paths:
   - `take_paths.py`

2. Split the paths of drones/not_drones and save them in folders:
    - `all_drones.csv` for drone
    - `all_not_drone.csv` for not_drone

3. Create labels when you have created `all_drones.csv` & `all_not_drone.csv`: 
    - `make_drone_labels_csv.py`
    - `make_not_drone_labels_csv.py`

4. Check if all data have labels:
    - `check_to_all_for_label_csv.py`

5. Copy the new data to the directory that contains every dataset:
    - `copy_data_to_dataset.py`

### Overfitting (optional)
#### Folder: `Classification_Training_Drone_Audio_Detection/overfitting`

1. **Change folder_name in file:**
   - `Classification_Training_Drone_Audio_Detection/overfitting/file_with_folder_name.json`
   - e.g., `"folder_name": "../data_mp3"`

2. Create folders for extra data:
   - `create_noise_folder_n_copy_files.py`

3. Delete mfcc from _noise folder to create a new one for each overfitting decision:
   - `delete_noise_mfcc.py`

4. Create `signal.csv` for each _noise:
   - `make_signal_noise.py`

5. Delete original signal because we have another from `make_signal_noise.py`:
   - `delete_original_signal.py`

6. Rename signal noise to signal:
   - `rename_signal_noise_to_signal.py`

7. Make `mfcc.csv` for all overfitting data:
   - `make_mfcc.py`

## Training  
#### Folder: `Create_Dataset_and_train`
   - `make_make_mfcc_labels.py`
     - output: `/mfcc_labels_{folder_name}.csv`
   - `saveh5.py`
     - output: `{folder_name}/train_dataset.h5`, `{folder_name}/val_dataset.h5`  , `{folder_name}/test_dataset.h5` . 
   - `loadh5.py`
     - output: `{folder_name}/trained_model_{folder_name}.keras`, `model12_{folder_name}.weights.h5')`, `{folder_name}/history_{folder_name}.pkl` .

- *if you want to create multiple dataset to compare them,make a new folder, copy `the mfcc_labels<name>.csv` and  then you can delete it some files or append new.
To delete can be done with:
   - `delete_lines_with_string.py` 

## Testing
## Create test dataset
#### Folder:  `create_test_dataset`
  - To get the dataset data in order to test you model:
    - `make_TEST_mfcc_labels.py` ,
      - appends data from e.g. `base_path = '../FINISHED_V7/ours_3'` to `output_file = 'mfcc_labels.csv'`
  - if you want to delete folders from `mfcc_labels.csv` , use : 
    - `delete_lines_with_string.py`,  set the variable `string_to_delete` to the string you want to delete
  - To create the `.h5` dataset :
    - `create_test_Dataset.v2.py`
      - using data from `labels_file_path = 'mfcc_labels.csv'`


## Get details for performance
#### Folder : `Compare`
  - To get the confusion matrix through a testing dataset :
    - `get_details.py`, add to the  variable `models_folder_names` , the folder location that model can found in `Create_Dataset_and_train` in folder.  
![My Image](./confuctionMatrix.png)



## Convert model to Raspberry Pi
