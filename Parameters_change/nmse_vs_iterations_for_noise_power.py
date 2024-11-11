import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO data with specified noise powers at a fixed SNR
def generate_thz_mimo_data_for_noise_powers(num_samples=10000, num_antennas=256, num_users=10, snr=5, noise_powers=None, save_file=False):
    if noise_powers is None:
        noise_powers = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
    
    # Generate the clean dataset (LoS + NLoS components)
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    # Container to store datasets for different noise power levels
    datasets = {}

    # Generate datasets for each noise power value
    for noise_power in noise_powers:
        noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
        noisy_channel_matrix = channel_matrix + noise

        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Store the datasets in a dictionary
        datasets[noise_power] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        # Optionally save the datasets to separate .npz files
        if save_file:
            filename = f'thz_mimo_dataset_snr_{snr}_noise_power_{noise_power:.0e}.npz'
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset for SNR {snr} dB with noise power {noise_power:.0e} saved as '{filename}'")

    return datasets

# Example usage
noise_powers = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5]
datasets = generate_thz_mimo_data_for_noise_powers(noise_powers=noise_powers, save_file=True)
