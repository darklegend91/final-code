import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO data with specified batch sizes at a constant SNR of 5 dB
def generate_thz_mimo_data_for_batch_sizes(num_samples=10000, num_antennas=256, num_users=10, snr=5, batch_sizes=None, save_file=False):
    if batch_sizes is None:
        batch_sizes = [32, 64, 128]
    
    # Generate the clean dataset (LoS + NLoS components)
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    # Calculate noise based on SNR
    noise_power = 1 / (10 ** (snr / 10))  # Convert SNR (dB) to noise power
    noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
    noisy_channel_matrix = channel_matrix + noise

    # Container to store datasets for different batch sizes
    datasets = {}

    # Generate datasets for each batch size
    for batch_size in batch_sizes:
        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Organize data in batches
        num_train_batches = len(X_train) // batch_size
        num_val_batches = len(X_val) // batch_size
        num_test_batches = len(X_test) // batch_size

        # Resize datasets to fit full batches
        X_train_batched = X_train[:num_train_batches * batch_size].reshape(num_train_batches, batch_size, num_antennas, num_users)
        y_train_batched = y_train[:num_train_batches * batch_size].reshape(num_train_batches, batch_size, num_antennas, num_users)

        X_val_batched = X_val[:num_val_batches * batch_size].reshape(num_val_batches, batch_size, num_antennas, num_users)
        y_val_batched = y_val[:num_val_batches * batch_size].reshape(num_val_batches, batch_size, num_antennas, num_users)

        X_test_batched = X_test[:num_test_batches * batch_size].reshape(num_test_batches, batch_size, num_antennas, num_users)
        y_test_batched = y_test[:num_test_batches * batch_size].reshape(num_test_batches, batch_size, num_antennas, num_users)

        # Store the datasets in a dictionary
        datasets[batch_size] = {
            'X_train': X_train_batched,
            'X_val': X_val_batched,
            'X_test': X_test_batched,
            'y_train': y_train_batched,
            'y_val': y_val_batched,
            'y_test': y_test_batched
        }

        # Optionally save the datasets to separate .npz files
        if save_file:
            filename = f'thz_mimo_dataset_snr_{snr}_batch_{batch_size}.npz'
            np.savez_compressed(filename, X_train=X_train_batched, X_val=X_val_batched, X_test=X_test_batched, 
                                y_train=y_train_batched, y_val=y_val_batched, y_test=y_test_batched)
            print(f"Dataset for SNR {snr} dB with batch size {batch_size} saved as '{filename}'")

    return datasets

# Example usage
batch_sizes = [32, 64, 128]
datasets = generate_thz_mimo_data_for_batch_sizes(batch_sizes=batch_sizes, save_file=True)
