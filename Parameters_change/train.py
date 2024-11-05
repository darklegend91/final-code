import os
import numpy as np
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Suppress TensorFlow warnings and force CPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors

# Function to add noise to the labels
def add_label_noise(y, noise_level=0.1):
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise

# Define linear and non-linear models with degradation
def linear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(np.prod(input_shape), activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

def nonlinear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(np.prod(input_shape), activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

# Train and save the models
def train_model(model, X_train, y_train, X_val, y_val, model_name='model'):
    checkpoint = ModelCheckpoint(f'{model_name}_best_weights.keras', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])
    model.save(f'{model_name}_final_model.keras')
    return history

# Main function to run training for different SNR levels
if __name__ == "__main__":
    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    noise_factor = 0.5

    for snr in snr_values:
        print(f"\nTraining models for SNR = {snr} dB")
        
        # Load the dataset for the current SNR
        data = np.load(f'thz_mimo_dataset_snr_{snr}.npz')
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Add noise to the input data
        X_train_noisy = X_train + noise_factor * np.random.randn(*X_train.shape)
        X_val_noisy = X_val + noise_factor * np.random.randn(*X_val.shape)
        
        # Add noise to labels
        y_train_noisy = add_label_noise(y_train, noise_level=0.5)
        y_val_noisy = add_label_noise(y_val, noise_level=0.5)
        
        # Train the linear model for the current SNR
        lin_model = linear_model(input_shape=X_train.shape[1:])
        train_model(lin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy, model_name=f'linear_model_snr_{snr}')
        
        # Train the non-linear model for the current SNR
        nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
        train_model(nonlin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy, model_name=f'nonlinear_model_snr_{snr}')
