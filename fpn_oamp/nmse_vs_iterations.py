import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('thz_mimo_dataset1.npz')
X_test = data['X_test']  # Noisy test set with additional noise added during generation
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
def save_results_to_excel(iterations, nmse_linear, nmse_nonlinear, filename='nmse_results.xlsx'):
    df = pd.DataFrame({
        'Iterations': iterations,
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# Evaluate across different iterations with added noise
iterations = [1, 5, 10, 15, 20]
nmse_linear = []
nmse_nonlinear = []

print("Evaluating models across different iterations with increasing noise:")
for iteration in iterations:
    # Add noise according to the iteration count
    noise = np.random.normal(0, 0.1, X_test.shape) * iteration  # Scaled noise per iteration
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
    print(f"Iteration: {iteration}")
    print(f"  Linear Model NMSE: {nmse_linear_value:.2f} dB")
    print(f"  Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")

# Plot the results
plt.plot(iterations, nmse_linear, 'o-', label='Linear Model NMSE')
plt.plot(iterations, nmse_nonlinear, 's-', label='Nonlinear Model NMSE')
plt.xlabel('Iterations (Noise Intensity)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid(True)
plt.title('NMSE vs Number of Iterations with Increasing Noise')

# Save the plot
plt.savefig('nmse_vs_iterations.png')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_iterations.png'")

# Save the NMSE results to an Excel file
save_results_to_excel(iterations, nmse_linear, nmse_nonlinear)
