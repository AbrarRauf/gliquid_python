from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


dump_dir = "all_dumps/gliq_manu_test7_linear/"
inter_doc = dump_dir + "ternary_Gliq_mps_final_linear_updated.xlsx"
meta_doc = dump_dir + "ternary_Gliq_meta_final_linear.json"

# Read the inter_df
inter_df = pd.read_excel(inter_doc)

# Ensure elements column is parsed as lists
inter_df['elements'] = inter_df['elements'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Create system keys from sorted elements
inter_df['system_key'] = inter_df['elements'].apply(lambda x: '-'.join(sorted(x)))

# Compute delta_T (positive = overpredicted, negative = underpredicted)
inter_df['delta_T'] = inter_df['gliq_melting_temp'] - inter_df['melting_point_k']

# Load metadata
with open(meta_doc, 'r') as f:
    meta_data = json.load(f)

# Filter and extract metadata
filtered_data = []

for idx, row in inter_df.iterrows():
    system_key = row['system_key']
    
    # Skip if system not in metadata
    if system_key not in meta_data:
        continue
    
    system_meta = meta_data[system_key]
    
    # Skip if "Contains predicted"
    if system_meta.get('Fit Type') == 'Contains predicted':
        continue
    
    # Extract ternary_meta
    ternary_meta = system_meta.get('ternary_meta', {})
    n_ternary_compounds = ternary_meta.get('n_ternary_compounds', None)
    deepest_formation_energy = ternary_meta.get('deepest_formation_energy', None)
    
    # Extract binary_L_params in fixed order: A-B, B-C, C-A
    binary_L_params = system_meta.get('binary_L_params', {})
    
    # Get sorted elements
    sorted_elems = sorted(row['elements'])
    
    # Define the three binaries in order: A-B, B-C, C-A
    binary_order = [
        (f"{sorted_elems[0]}-{sorted_elems[1]}", "AB"),  # A-B
        (f"{sorted_elems[1]}-{sorted_elems[2]}", "BC"),  # B-C
        (f"{sorted_elems[2]}-{sorted_elems[0]}", "CA")   # C-A
    ]
    
    # Extract L parameters: each binary has 4 params [L0_a, L0_b, L1_a, L1_b]
    L_params_ordered = {}
    for bin_key, prefix in binary_order:
        if bin_key in binary_L_params:
            params = binary_L_params[bin_key]
            # params = [L0_a, L0_b, L1_a, L1_b]
            L_params_ordered[f'{prefix}_L0_a'] = params[0] if len(params) > 0 else 0
            L_params_ordered[f'{prefix}_L0_b'] = params[1] if len(params) > 1 else 0
            L_params_ordered[f'{prefix}_L1_a'] = params[2] if len(params) > 2 else 0
            L_params_ordered[f'{prefix}_L1_b'] = params[3] if len(params) > 3 else 0
        else:
            # If binary not found, fill with zeros
            L_params_ordered[f'{prefix}_L0_a'] = 0
            L_params_ordered[f'{prefix}_L0_b'] = 0
            L_params_ordered[f'{prefix}_L1_a'] = 0
            L_params_ordered[f'{prefix}_L1_b'] = 0
    
    # Create entry
    entry = {
        'reduced_formula': row['reduced_formula'],
        'melting_point_k': row['melting_point_k'],
        'elements': row['elements'],
        'gliq_melting_temp': row['gliq_melting_temp'],
        'type': row['type'],
        'system_key': system_key,
        'delta_T': row['delta_T'],
        'n_ternary_compounds': n_ternary_compounds,
        'deepest_formation_energy': deepest_formation_energy,
    }
    entry.update(L_params_ordered)
    
    filtered_data.append(entry)

# Create filtered dataframe
filtered_df = pd.DataFrame(filtered_data)

print(f"Total systems after filtering: {len(filtered_df)}")
print(f"Delta_T range: [{filtered_df['delta_T'].min():.1f}, {filtered_df['delta_T'].max():.1f}]")

# Split into quartiles based on delta_T
filtered_df['quartile'] = pd.qcut(filtered_df['delta_T'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Stratified sampling function
def stratified_sample(group, n_samples=5):
    """
    Sample systems from a quartile with stratification on:
    - Sign of delta_T (over/under predicted)
    - n_ternary_compounds diversity
    - deepest_formation_energy diversity
    - Binary L parameter diversity
    """
    
    # Ensure we have enough samples
    if len(group) <= n_samples:
        return group
    
    # Add sign of delta_T as a feature
    group = group.copy()
    group['delta_T_sign'] = np.sign(group['delta_T'])
    
    # Get features for stratification
    feature_cols = ['n_ternary_compounds', 'deepest_formation_energy', 'delta_T_sign']
    
    # Add binary L parameter columns
    L_param_cols = [col for col in group.columns if col.endswith(('_L0', '_L1', '_L2', '_L3'))]
    feature_cols.extend(L_param_cols)
    
    # Prepare features
    X = group[feature_cols].fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use k-means clustering to find diverse samples
    n_clusters = min(n_samples, len(group))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    group['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Sample one from each cluster, prioritizing those closest to cluster centers
    sampled_indices = []
    for cluster_id in range(n_clusters):
        cluster_group = group[group['cluster'] == cluster_id]
        if len(cluster_group) > 0:
            # Find the point closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_X = X_scaled[group['cluster'] == cluster_id]
            distances = np.linalg.norm(cluster_X - cluster_center, axis=1)
            closest_idx = cluster_group.index[np.argmin(distances)]
            sampled_indices.append(closest_idx)
    
    return group.loc[sampled_indices]

# Sample from each quartile
validation_set = []

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    quartile_group = filtered_df[filtered_df['quartile'] == quartile]
    print(f"\n{quartile}: {len(quartile_group)} systems, delta_T range: [{quartile_group['delta_T'].min():.1f}, {quartile_group['delta_T'].max():.1f}]")
    
    # Sample 4-6 systems (use 5 as default)
    sampled = stratified_sample(quartile_group, n_samples=5)
    print(f"  Sampled {len(sampled)} systems")
    
    validation_set.append(sampled)

# Combine all sampled systems
validation_df = pd.concat(validation_set, ignore_index=True)

# Drop temporary columns
validation_df = validation_df.drop(columns=['cluster', 'delta_T_sign'], errors='ignore')

print(f"\nTotal validation set size: {len(validation_df)}")
print(f"Distribution by quartile: {validation_df['quartile'].value_counts().sort_index()}")
print(f"Distribution by type: {validation_df['type'].value_counts()}")
print(f"Distribution by delta_T sign: Over={sum(validation_df['delta_T'] > 0)}, Under={sum(validation_df['delta_T'] < 0)}")

# Save to Excel
output_path = dump_dir + "validation_set_stratified.xlsx"
validation_df.to_excel(output_path, index=False)
print(f"\nValidation set saved to: {output_path}")

