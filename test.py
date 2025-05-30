import os
import numpy as np

input_folder = "/workspace/preprocessed_data_resized"

# List all .npy files in the folder
files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

# Loop through and print shape of each file
for i, file in enumerate(files):
    filepath = os.path.join(input_folder, file)
    arr = np.load(filepath)
    print(f"{file}: shape = {arr.shape}")

    if i >= 4:  # Just print the first 5 files
        break
