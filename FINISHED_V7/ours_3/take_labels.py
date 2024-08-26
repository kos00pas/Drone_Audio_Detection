import h5py

# Open the HDF5 file
file_path = 'ours_3_test_dataset.h5'

with h5py.File(file_path, 'r') as f:
    # List all.txt groups and datasets
    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    f.visititems(print_attrs)
