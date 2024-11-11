import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO data with a fixed SNR of 5 dB and variable user counts
def generate_thz_mimo_data_for_fixed_snr(num_samples=10000, num_antennas=256, user_counts=None, save_file=False):
    if user_counts is None:
        user_counts = [2, 4, 6, 8, 10]  # Specify the number of users

    # Fixed SNR value
    snr = 5
    noise_power = 1 / (10 ** (snr / 10))  # Convert SNR (dB) to noise power

    # Generate the clean dataset (LoS + NLoS components)
    los_component = np.random.randn(num_samples, num_antennas, max(user_counts))
    nlos_component = np.random.randn(num_samples, num_antennas, max(user_counts)) * 0.1
    channel_matrix = los_component + nlos_component

    # Container to store datasets for different user counts
    datasets = {}

    # Generate datasets for each user count
    for num_users in user_counts:
        noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
        noisy_channel_matrix = channel_matrix[:, :, :num_users] + noise

        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix[:, :, :num_users], test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Store the datasets in a dictionary
        datasets[num_users] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        # Optionally save the datasets to separate .npz files
        if save_file:
            filename = f'thz_mimo_dataset_snr_{snr}_users_{num_users}.npz'
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset for SNR {snr} dB and {num_users} users saved as '{filename}'")

    return datasets

# Example usage
user_counts = [2, 4, 6, 8, 10]
datasets = generate_thz_mimo_data_for_fixed_snr(user_counts=user_counts, save_file=True)
