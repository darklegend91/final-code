import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 5 * np.log10(mse / norm_factor)

# Define a function to save NMSE results to an Excel file
def save_results_to_excel(user_configs, iterations, nmse_linear, nmse_nonlinear, filename='nmse_results_users_iterations.xlsx'):
    # Calculate correct lengths and trim as needed
    min_len = min(len(nmse_linear), len(nmse_nonlinear), len(user_configs) * len(iterations))
    nmse_linear, nmse_nonlinear = nmse_linear[:min_len], nmse_nonlinear[:min_len]
    expanded_user_configs = np.repeat(user_configs, len(iterations))[:min_len]  # Repeat each user config for all iterations
    expanded_iterations = (iterations * len(user_configs))[:min_len]  # Expand iterations to match users

    # Create the DataFrame
    df = pd.DataFrame({
        'Number of Users': expanded_user_configs,
        'Iterations': expanded_iterations,
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# Fixed SNR value and user configurations to evaluate
snr_value = 5
user_configs = [2, 4, 6, 8, 10]  # Number of users
iterations = list(range(1, 16))  # 1 to 15 iterations

# Store NMSE values for all user configurations and iterations
nmse_linear_all = []
nmse_nonlinear_all = []

print("Evaluating models across different user configurations and iterations:")
for num_users in user_configs:
    print(f"\nEvaluating for Number of Users: {num_users}")
    
    # Load the dataset for the current user configuration at SNR = 5 dB
    data_path = f'thz_mimo_dataset_snr_{snr_value}_users_{num_users}.npz'
    if not os.path.exists(data_path):
        print(f"Data file for {num_users} users not found, skipping.")
        continue
    
    data = np.load(data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load the trained models for the current user configuration
    linear_model_path = f'linear_model_users_{num_users}_final_model.keras'
    nonlinear_model_path = f'nonlinear_model_users_{num_users}_final_model.keras'
    
    if not os.path.exists(linear_model_path) or not os.path.exists(nonlinear_model_path):
        print(f"Model files for {num_users} users not found, skipping.")
        continue

    linear_model = load_model(linear_model_path)
    nonlinear_model = load_model(nonlinear_model_path)
    
    # Initialize lists to store NMSE values for each iteration for this user configuration
    nmse_linear = []
    nmse_nonlinear = []
    
    for iteration in iterations:
        # Add noise according to the iteration count
        noise_power = 10 ** (-snr_value / 10)
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
    
    # Append the results for this user configuration
    nmse_linear_all.extend([-val for val in nmse_linear])  # Negate NMSE values for plotting
    nmse_nonlinear_all.extend([-val for val in nmse_nonlinear])  # Negate NMSE values for plotting
    
    # Plot NMSE for this user configuration
    plt.plot(iterations, nmse_linear, 'o-', label=f'Linear Model NMSE (Users={num_users})')
    plt.plot(iterations, nmse_nonlinear, 's-', label=f'Nonlinear Model NMSE (Users={num_users})')

# Configure the plot
plt.xlabel('Iterations (Noise Intensity)')
plt.ylabel('NMSE (dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.title('NMSE vs Iterations for Different User Configurations at SNR 5 dB')

# Save the plot
plt.savefig('nmse_vs_iterations_for_users_snr_5db.png', bbox_inches='tight')
plt.show()
print("Chart saved as 'nmse_vs_iterations_for_users_snr_5db.png'")

# Save the NMSE results to an Excel file
save_results_to_excel(user_configs, iterations, nmse_linear_all, nmse_nonlinear_all)


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.metrics import mean_squared_error
# import os

# # Define a function to compute NMSE (Normalized Mean Squared Error)
# def compute_nmse(y_true, y_pred):
#     mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
#     norm_factor = np.linalg.norm(y_true.flatten()) ** 2
#     return 5 * np.log10(mse / norm_factor)

# # Define a function to save NMSE results to an Excel file
# def save_results_to_excel(user_configs, iterations, nmse_linear, nmse_nonlinear, filename='nmse_results_users_iterations.xlsx'):
#     df = pd.DataFrame({
#         'Number of Users': np.repeat(user_configs, len(iterations)),
#         'Iterations': iterations * len(user_configs),
#         'Linear Model NMSE (dB)': nmse_linear,
#         'Nonlinear Model NMSE (dB)': nmse_nonlinear
#     })
#     df.to_excel(filename, index=False)
#     print(f"Results saved to '{filename}'")

# # Fixed SNR value and user configurations to evaluate
# snr_value = 5
# user_configs = [2, 4, 6, 8, 10]  # Number of users
# iterations = list(range(1, 16))  # 1 to 15 iterations

# # Store NMSE values for all user configurations and iterations
# nmse_linear_all = []
# nmse_nonlinear_all = []

# print("Evaluating models across different user configurations and iterations:")
# for num_users in user_configs:
#     print(f"\nEvaluating for Number of Users: {num_users}")
    
#     # Load the dataset for the current user configuration at SNR = 5 dB
#     data = np.load(f'thz_mimo_dataset_snr_{snr_value}_users_{num_users}.npz')
#     X_test = data['X_test']
#     y_test = data['y_test']
    
#     # Load the trained models for the current user configuration
#     linear_model_path = f'linear_model_users_{num_users}_final_model.keras'
#     nonlinear_model_path = f'nonlinear_model_users_{num_users}_final_model.keras'
    
#     if not os.path.exists(linear_model_path) or not os.path.exists(nonlinear_model_path):
#         print(f"Model files for {num_users} users not found.")
#         continue

#     linear_model = load_model(linear_model_path)
#     nonlinear_model = load_model(nonlinear_model_path)
    
#     # Initialize lists to store NMSE values for each iteration for this user configuration
#     nmse_linear = []
#     nmse_nonlinear = []
    
#     for iteration in iterations:
#         # Add noise according to the iteration count
#         noise_power = 10 ** (-snr_value / 10)
#         noise = np.random.normal(0, np.sqrt(noise_power), X_test.shape) * iteration
#         X_test_noisy = X_test + noise  # Adding noise to the test data
        
#         # Make predictions
#         y_pred_linear = linear_model.predict(X_test_noisy)
#         y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
        
#         # Compute NMSE for both models
#         nmse_linear_value = compute_nmse(y_test, y_pred_linear)
#         nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)
#         nmse_linear.append(nmse_linear_value)
#         nmse_nonlinear.append(nmse_nonlinear_value)
        
#         # Print NMSE results for the current iteration
#         print(f"  Iteration: {iteration}")
#         print(f"    Linear Model NMSE: {nmse_linear_value:.2f} dB")
#         print(f"    Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")
    
#     # Append the results for this user configuration
#     nmse_linear_all.extend([-val for val in nmse_linear])  # Negate NMSE values for plotting
#     nmse_nonlinear_all.extend([-val for val in nmse_nonlinear])  # Negate NMSE values for plotting
    
#     # Plot NMSE for this user configuration
#     plt.plot(iterations, nmse_linear_all[-len(nmse_linear):], 'o-', label=f'Linear Model NMSE (Users={num_users})')
#     plt.plot(iterations, nmse_nonlinear_all[-len(nmse_nonlinear):], 's-', label=f'Nonlinear Model NMSE (Users={num_users})')

# # Configure the plot
# plt.xlabel('Iterations (Noise Intensity)')
# plt.ylabel('NMSE (dB)')
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.grid(True)
# plt.title('NMSE vs Iterations for Different User Configurations at SNR 5 dB')

# # Save the plot
# plt.savefig('nmse_vs_iterations_for_users_snr_5db.png', bbox_inches='tight')
# plt.show()
# print("Chart saved as 'nmse_vs_iterations_for_users_snr_5db.png'")

# # Save the NMSE results to an Excel file
# save_results_to_excel(user_configs, iterations, nmse_linear_all, nmse_nonlinear_all)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

# # Function to calculate NMSE
# def calculate_nmse(y_true, y_pred):
#     return np.mean((np.linalg.norm(y_true - y_pred, axis=1) ** 2) / (np.linalg.norm(y_true, axis=1) ** 2))

# # Define the configurations
# iterations = np.arange(1, 101)  # Adjust based on actual iteration count
# user_configs = [2, 4, 6, 8, 10]

# # Load the trained model (adjust file paths as needed)
# model = load_model('/workspaces/final-code/new_folder/dataset')  # Replace with your actual model path

# # Dictionary to store NMSE for each user count
# nmse_data = {}

# # Compute NMSE for each user configuration
# for user in user_configs:
#     filename = f'thz_mimo_dataset_users_{user}.npz'
#     data = np.load(filename)
#     X_test = data['X_test']
#     y_test = data['y_test']

#     # Calculate NMSE over the defined iterations
#     nmse_values = []
#     for _ in iterations:
#         y_pred = model.predict(X_test)  # Generate predictions using the model
#         nmse = calculate_nmse(y_test, y_pred)
#         nmse_values.append(nmse)

#     nmse_data[user] = nmse_values

# # Save NMSE data to an Excel file
# def save_nmse_to_excel(nmse_data, filename='nmse_vs_iterations.xlsx', sheet_name='NMSE Results'):
#     df = pd.DataFrame(nmse_data)
#     df['Iterations'] = iterations
#     df = df.set_index('Iterations')
#     df.to_excel(filename, index=True, sheet_name=sheet_name)
#     print(f"Results saved to '{filename}' in sheet '{sheet_name}'")

# # Save the results
# save_nmse_to_excel(nmse_data)

# # Plot NMSE vs Iterations
# plt.figure(figsize=(10, 6))
# for user in user_configs:
#     plt.plot(iterations, nmse_data[user], label=f'Users: {user}', marker='o')

# plt.xlabel('Iterations')
# plt.ylabel('Normalized Mean Square Error (NMSE)')
# plt.title('NMSE vs Iterations for Different Number of Users')
# plt.legend()
# plt.grid(True)
# plt.savefig('nmse_vs_iterations.png')
# plt.show()
# print("Chart saved as 'nmse_vs_iterations.png'")


# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Sample iteration range; adjust based on your actual training iterations
# # iterations = np.arange(1, 101)  # Assuming 100 iterations, update if needed
# # user_configs = [2, 4, 6, 8, 10]

# # # Initialize a dictionary to store NMSE values for each user configuration
# # nmse_data = {}

# # # Load NMSE data from each user's file
# # for user in user_configs:
# #     # Load the dataset file for the current user count
# #     filename = f'thz_mimo_dataset_snr_5_users_{user}.npz'
# #     data = np.load(filename)
    
# #     # Assuming 'nmse_values' array is saved in each file for each iteration
# #     # Adjust key names if your data structure is different
# #     nmse_data[user] = data['nmse_values']  # Replace 'nmse_values' with the correct key if necessary

# # # Save NMSE results to an Excel file
# # def save_nmse_to_excel(nmse_data, filename='nmse_vs_iterations.xlsx', sheet_name='NMSE Results'):
# #     df = pd.DataFrame(nmse_data)
# #     df['Iterations'] = iterations
# #     df = df.set_index('Iterations')
# #     df.to_excel(filename, index=True, sheet_name=sheet_name)
# #     print(f"Results saved to '{filename}' in sheet '{sheet_name}'")

# # # Save NMSE data to Excel
# # save_nmse_to_excel(nmse_data)

# # # Plot the NMSE vs Iterations
# # plt.figure(figsize=(10, 6))
# # for user in user_configs:
# #     plt.plot(iterations, nmse_data[user], label=f'Users: {user}', marker='o')

# # plt.xlabel('Iterations')
# # plt.ylabel('Normalized Mean Square Error (NMSE)')
# # plt.title('NMSE vs Iterations for Different Number of Users')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('nmse_vs_iterations.png')  # Save the chart
# # plt.show()  # Show the plot
# # print("Chart saved as 'nmse_vs_iterations.png'")
