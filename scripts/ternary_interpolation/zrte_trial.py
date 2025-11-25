import numpy as np
import ast
import pandas as pd

# Data as string (you can replace this with file reading if needed)
data_lines = """275	293.65	[0.78666667, 0.11666667]	[0.921, 0.077]
375	376.15	[0.25238095, 0.41380952]	[0.202, 0.259]
275	290.15	[0.07538462, 0.09923077]	[0.0664, 0.0321]
415	402.5	[0.48871795, 0.27141026]	[0.42, 0.39]
425	436.15	[0.56709677, 0.12846774]	[0.71, 0.04]
350	369.15	[0.70428571, 0.01571429]	[0.32, 0.16]""".strip().split('\n')

# Parse the data
temp1_list = []
temp2_list = []
comp1_list = []
comp2_list = []

for line in data_lines:
    parts = line.split('\t')
    temp1 = float(parts[0])
    temp2 = float(parts[1])
    comp1 = ast.literal_eval(parts[2])  # Convert string representation of list to actual list
    comp2 = ast.literal_eval(parts[3])
    
    temp1_list.append(temp1)
    temp2_list.append(temp2)
    comp1_list.append(comp1)
    comp2_list.append(comp2)

# Convert to numpy arrays for easier computation
temp1_array = np.array(temp1_list)
temp2_array = np.array(temp2_list)
comp1_array = np.array(comp1_list)
comp2_array = np.array(comp2_list)

# Compute MAPE for temperature (Mean Absolute Percentage Error)
delta_T = np.abs(temp1_array - temp2_array)
mape_T_individual = (delta_T / np.abs(temp1_array)) * 100  # Individual MAPE values
mape_T = np.mean(mape_T_individual)  # Overall MAPE

# Compute individual squared errors for temperature (individual RMSE contributions)
temp_squared_errors = (temp1_array - temp2_array)**2
individual_rmse_T = np.sqrt(temp_squared_errors)

# Compute RMSE for temperature differences
rmse_T = np.sqrt(np.mean(temp_squared_errors))

# Compute delta_c (2D distances between compositions)
delta_c = np.sqrt(np.sum((comp1_array - comp2_array)**2, axis=1))

# Compute MAPE for compositions (using magnitude of composition vectors as reference)
comp1_magnitude = np.sqrt(np.sum(comp1_array**2, axis=1))  # Magnitude of comp1 vectors
# Avoid division by zero by using a small epsilon where magnitude is very small
comp1_magnitude = np.where(comp1_magnitude < 1e-10, 1e-10, comp1_magnitude)
mape_c_individual = (delta_c / comp1_magnitude) * 100  # Individual MAPE values
mape_c = np.mean(mape_c_individual)  # Overall MAPE

# Individual RMSE contributions for composition (same as delta_c since it's already a distance)
individual_rmse_c = delta_c

# Compute RMSE for composition differences (treating as 2D error)
rmse_c = np.sqrt(np.mean(delta_c**2))

# Print results
print("Individual temperature differences (absolute):", delta_T)
print("Individual temperature MAPE (%):", mape_T_individual)
print("Individual temperature RMSE contributions:", individual_rmse_T)
print("Individual composition distances (delta_c):", delta_c)
print("Individual composition MAPE (%):", mape_c_individual)
print("Individual composition RMSE contributions:", individual_rmse_c)

print(f"\nOverall Statistics:")
print(f"Temperature MAPE: {mape_T:.4f}%")
print(f"Composition MAPE: {mape_c:.4f}%")
print(f"RMSE Temperature: {rmse_T:.4f}")
print(f"RMSE Composition: {rmse_c:.4f}")

print(f"\nRow-by-Row RMSE Analysis:")
for i in range(len(temp1_list)):
    print(f"Row {i+1}: Temp RMSE = {individual_rmse_T[i]:.4f}, Comp RMSE = {individual_rmse_c[i]:.4f}")

# Compute coefficient of variation (relative standard deviation) for percentage metrics
cv_T = (np.std(mape_T_individual) / mape_T) * 100  # CV of temperature MAPE
cv_c = (np.std(mape_c_individual) / mape_c) * 100  # CV of composition MAPE

# Compute relative standard deviation for absolute errors as percentage of mean
rel_std_T = (np.std(delta_T) / np.mean(delta_T)) * 100  # Relative std dev for temp differences
rel_std_c = (np.std(delta_c) / np.mean(delta_c)) * 100  # Relative std dev for comp distances

# Additional statistics
print(f"\nDetailed Statistics:")
print(f"Temperature - MAPE: {mape_T:.4f}%, RMSE: {rmse_T:.4f}, Min: {np.min(delta_T):.4f}, Max: {np.max(delta_T):.4f}")
print(f"  Absolute Std: {np.std(delta_T):.4f}, Relative Std: {rel_std_T:.4f}%, MAPE CV: {cv_T:.4f}%")
print(f"Composition - MAPE: {mape_c:.4f}%, RMSE: {rmse_c:.4f}, Min: {np.min(delta_c):.4f}, Max: {np.max(delta_c):.4f}")
print(f"  Absolute Std: {np.std(delta_c):.4f}, Relative Std: {rel_std_c:.4f}%, MAPE CV: {cv_c:.4f}%")
print(f"Temperature MAPE range: {np.min(mape_T_individual):.4f}% to {np.max(mape_T_individual):.4f}%")
print(f"Composition MAPE range: {np.min(mape_c_individual):.4f}% to {np.max(mape_c_individual):.4f}%")


inter_path = 'all_dumps/gliq_manu_test4/ternary_Gliq_mps_final_linear.xlsx'

# Load Excel file into pandas DataFrame
print("="*80)
print("EXCEL FILE ANALYSIS")
print("="*80)

df = pd.read_excel(inter_path)
print(f"Excel file loaded successfully. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check if required columns exist
required_cols = ["melting_point_k", "gliq_melting_temp"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    print("Available columns:", list(df.columns))
else:
    print(f"Found required columns: {required_cols}")
    
    # Extract temperature data
    melting_point_k = df["melting_point_k"].dropna()  # Remove NaN values
    gliq_melting_temp = df["gliq_melting_temp"].dropna()  # Remove NaN values
    
    # Align datasets (use rows where both columns have values)
    valid_indices = df[["melting_point_k", "gliq_melting_temp"]].dropna().index
    melting_point_aligned = df.loc[valid_indices, "melting_point_k"].values
    gliq_melting_aligned = df.loc[valid_indices, "gliq_melting_temp"].values
    
    print(f"Valid temperature pairs: {len(melting_point_aligned)}")
    print(f"melting_point_k range: {melting_point_aligned.min():.2f} to {melting_point_aligned.max():.2f}")
    print(f"gliq_melting_temp range: {gliq_melting_aligned.min():.2f} to {gliq_melting_aligned.max():.2f}")
    
    # Compute MAPE using melting_point_k as reference (actual values)
    temp_differences = np.abs(gliq_melting_aligned - melting_point_aligned)
    excel_mape_individual = (temp_differences / np.abs(melting_point_aligned)) * 100
    excel_mape = np.mean(excel_mape_individual)
    
    # Compute relative standard deviation
    excel_rel_std = (np.std(temp_differences) / np.mean(temp_differences)) * 100
    
    # Compute MAPE coefficient of variation
    excel_mape_cv = (np.std(excel_mape_individual) / excel_mape) * 100
    
    # Compute RMSE for comparison
    excel_rmse = np.sqrt(np.mean((gliq_melting_aligned - melting_point_aligned)**2))
    
    # Print Excel analysis results
    print(f"\nEXCEL FILE TEMPERATURE ANALYSIS:")
    print(f"MAPE (gliq_melting_temp vs melting_point_k): {excel_mape:.4f}%")
    print(f"RMSE: {excel_rmse:.4f} K")
    print(f"Absolute differences - Mean: {np.mean(temp_differences):.4f}, Std: {np.std(temp_differences):.4f}")
    print(f"Relative Std Dev: {excel_rel_std:.4f}%")
    print(f"MAPE Coefficient of Variation: {excel_mape_cv:.4f}%")
    print(f"MAPE range: {np.min(excel_mape_individual):.4f}% to {np.max(excel_mape_individual):.4f}%")
    print(f"Temperature differences range: {np.min(temp_differences):.4f} to {np.max(temp_differences):.4f} K")

