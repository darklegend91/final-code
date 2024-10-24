import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('thz_mimo_dataset1.npz')
X_test = data['X_test']  # Noisy test set
y_test = data['y_test']

# Load the trained linear and non-linear models
linear_model = load_model('linear_model_final_model.keras')
nonlinear_model = load_model('nonlinear_model_final_model.keras')

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 5 * np.log10(mse / norm_factor)

# Define a function to save results to an Excel file
def save_results_to_excel(snr_levels, nmse_linear, nmse_nonlinear, filename='nmse_results.xlsx'):
    df = pd.DataFrame({
        'SNR (dB)': snr_levels,
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# Evaluate across different SNR levels (Signal-to-Noise Ratio in dB)
snr_levels = [-20, -15, -10 , -5 , 0, 5, 10, 15, 20]  # SNR values in dB
nmse_linear = []
nmse_nonlinear = []

print("Evaluating models across different SNR levels:")
for snr in snr_levels:
    # Add noise according to SNR level (converting SNR to noise power)
    noise_power = 10 ** (-snr / 20)
    noise = np.random.normal(0, noise_power, X_test.shape)
    X_test_noisy = X_test + noise
    
    # Make predictions
    y_pred_linear = linear_model.predict(X_test_noisy)
    y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
    
    # Compute NMSE
    nmse_linear_value = compute_nmse(y_test, y_pred_linear)
    nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)
    nmse_linear.append(nmse_linear_value)
    nmse_nonlinear.append(nmse_nonlinear_value)
    
    # Print NMSE results for the current SNR level
    print(f"SNR: {snr} dB")
    print(f"  Linear Model NMSE: {nmse_linear_value:.2f} dB")
    print(f"  Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")

# Save the results to an Excel file
save_results_to_excel(snr_levels, nmse_linear, nmse_nonlinear)

# Plot the results
plt.plot(snr_levels, nmse_linear, 'o-', label='Linear Model NMSE')
plt.plot(snr_levels, nmse_nonlinear, 's-', label='Nonlinear Model NMSE')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid(True)
plt.title('NMSE vs SNR')

# Save the plot
plt.savefig('nmse_vs_snr.png')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_snr.png'")
