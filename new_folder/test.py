import numpy as np

# Replace '2' with any user configuration file you want to inspect
filename = 'thz_mimo_dataset_snr_5_users_2.npz' 
data = np.load(filename)

print("Keys in the .npz file:", list(data.keys()))
