import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('thz_mimo_dataset1.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load the trained models
linear_model = load_model('linear_model_final_model.keras')
nonlinear_model = load_model('nonlinear_model_final_model.keras')

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 5 * np.log10(mse / norm_factor)

# Define a function to save NMSE results to an Excel file
def save_results_to_excel(snr_values, iterations, nmse_linear, nmse_nonlinear, filename='nmse_results_snr_iterations.xlsx'):
    df = pd.DataFrame({
        'SNR (dB)': np.repeat(snr_values, len(iterations)),
        'Iterations': iterations * len(snr_values),
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# SNR values and iterations to evaluate
snr_values = [-10, -5, 0, 5, 10, 15, 20]
iterations = list(range(1, 16))  # 1 to 15 iterations

# Store NMSE values for all SNRs and iterations
nmse_linear_all = []
nmse_nonlinear_all = []

print("Evaluating models across different SNR values and iterations:")
for snr in snr_values:
    print(f"\nEvaluating for SNR: {snr} dB")
    noise_power = 10 ** (-snr / 10)
    nmse_linear = []
    nmse_nonlinear = []
    
    for iteration in iterations:
        # Add noise according to the iteration count and SNR
        noise = np.random.normal(0, np.sqrt(noise_power), X_test.shape) * iteration
        X_test_noisy = X_test + noise  # Adding noise to the test data
        
        # Make predictions
        y_pred_linear = linear_model.predict(X_test_noisy)
        y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
        
        # Compute NMSE for both models
        nmse_linear_value = compute_nmse(y_test, y_pred_linear)
        nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)
        nmse_linear.append(nmse_linear_value)
        nmse_nonlinear.append(nmse_nonlinear_value)
        
        # Print NMSE results for the current iteration
        print(f"  Iteration: {iteration}")
        print(f"    Linear Model NMSE: {nmse_linear_value:.2f} dB")
        print(f"    Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")
    
    # Append the results for this SNR
    nmse_linear_all.extend(nmse_linear)
    nmse_nonlinear_all.extend(nmse_nonlinear)
    
    # Plot NMSE for this SNR
    plt.plot(iterations, nmse_linear, 'o-', label=f'Linear Model NMSE (SNR={snr} dB)')
    plt.plot(iterations, nmse_nonlinear, 's-', label=f'Nonlinear Model NMSE (SNR={snr} dB)')

# Configure the plot
plt.xlabel('Iterations (Noise Intensity)')
plt.ylabel('NMSE (dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.title('NMSE vs Iterations for Different SNR Levels')

# Save the plot
plt.savefig('nmse_vs_iterations_for_snr.png', bbox_inches='tight')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_iterations_for_snr.png'")

# Save the NMSE results to an Excel file
save_results_to_excel(snr_values, iterations, nmse_linear_all, nmse_nonlinear_all)
