import pandas as pd 
from pymatgen.core import Composition


ternary_df = pd.read_excel("data/ternary_dft_data/ternary_im_melting_points.xlsx")
to_modify_df = pd.read_excel("data/ternary_dft_data/ternary_Gliq_mps_final_linear.xlsx")
mapp_df = pd.read_csv("data/ternary_dft_data/output.csv")

print(ternary_df)
print(f"\nTotal entries: {len(ternary_df)}")
print(f"Unique formulas: {ternary_df['formula'].nunique()}")

# Preprocess formulas: remove phase labels and convert to reduced formulas
def clean_and_reduce_formula(formula_str):
    # Split by space and take only the first part (removes phase labels)
    formula_clean = formula_str.split()[0]
    
    # Use pymatgen to get reduced formula
    try:
        comp = Composition(formula_clean)
        return comp.reduced_formula
    except Exception as e:
        print(f"Warning: Could not parse formula '{formula_str}': {e}")
        return None

# Apply preprocessing to ternary_df
ternary_df['reduced_formula'] = ternary_df['formula'].apply(clean_and_reduce_formula)

# Remove rows where formula parsing failed
ternary_df = ternary_df.dropna(subset=['reduced_formula'])

print(f"\nAfter preprocessing:")
print(f"Total entries: {len(ternary_df)}")
print(f"Unique reduced formulas: {ternary_df['reduced_formula'].nunique()}")
print("\nSample of original vs reduced formulas:")
print(ternary_df[['formula', 'reduced_formula']].head(10))

# IQR-based outlier removal and averaging
def remove_outliers_iqr(group):
    """
    Remove outliers using IQR method and return mean of remaining values.
    
    For groups with < 3 measurements, return the mean without filtering.
    """
    if len(group) < 3:
        return group.mean()
    
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered = group[(group >= lower_bound) & (group <= upper_bound)]
    
    # If all values are filtered out (shouldn't happen with IQR), return median
    return filtered.mean() if len(filtered) > 0 else group.median()

# Group by formula and apply IQR filtering
grouped = ternary_df.groupby('reduced_formula')['melting_point_k']
avg_temps = grouped.apply(remove_outliers_iqr).reset_index()
avg_temps.columns = ['reduced_formula', 'avg_melting_point_k']

# Rebuild ternary element lists for each reduced formula so downstream files
# keep the same schema expected by batch scripts.
def reduced_formula_to_elements(reduced_formula):
    comp = Composition(reduced_formula)
    return sorted([el.symbol for el in comp.elements])

avg_temps['elements'] = avg_temps['reduced_formula'].apply(reduced_formula_to_elements)

# Round to 3 decimal places
avg_temps['avg_melting_point_k'] = avg_temps['avg_melting_point_k'].round(3)

# Keep output column order compatible with prior filtered files.
avg_temps = avg_temps[['reduced_formula', 'elements', 'avg_melting_point_k']]

print(f"\nFiltered results: {len(avg_temps)} unique reduced formulas")
print(avg_temps.head(10))

# Save to Excel
output_file = "data/ternary_dft_data/ternary_im_filtered_IQR.xlsx"
avg_temps.to_excel(output_file, index=False)
print(f"\nSaved filtered data to: {output_file}")

# Update to_modify_df with IQR-filtered temperatures from ternary_df
print("\n" + "="*60)
print("Updating to_modify_df with IQR-filtered temperatures")
print("="*60)

# Check if all formulas in to_modify_df exist in avg_temps
to_modify_formulas = set(to_modify_df['reduced_formula'].unique())
avg_temps_formulas = set(avg_temps['reduced_formula'].unique())

missing_formulas = to_modify_formulas - avg_temps_formulas

if missing_formulas:
    print(f"\nWARNING: {len(missing_formulas)} formulas in to_modify_df are NOT in ternary_df:")
    print(f"Missing formulas: {sorted(missing_formulas)}")
else:
    print(f"\nAll {len(to_modify_formulas)} formulas in to_modify_df exist in ternary_df ✓")

# Create a mapping from reduced_formula to avg_melting_point_k
temp_mapping = dict(zip(avg_temps['reduced_formula'], avg_temps['avg_melting_point_k']))

# Update melting_point_k in to_modify_df
to_modify_df['melting_point_k'] = to_modify_df['reduced_formula'].map(temp_mapping)

# Check for any NaN values (formulas that couldn't be mapped)
unmapped_count = to_modify_df['melting_point_k'].isna().sum()
if unmapped_count > 0:
    print(f"\nWARNING: {unmapped_count} rows could not be mapped to temperatures")
else:
    print(f"\nSuccessfully updated all {len(to_modify_df)} rows with IQR-filtered temperatures ✓")

# Save updated to_modify_df
output_file_modified = "data/ternary_dft_data/ternary_Gliq_mps_final_linear_updated.xlsx"
to_modify_df.to_excel(output_file_modified, index=False)
print(f"\nSaved updated data to: {output_file_modified}")

# Update mapp_df with IQR-filtered temperatures from ternary_df
print("\n" + "="*60)
print("Updating mapp_df with IQR-filtered temperatures")
print("="*60)

# Check if all formulas in mapp_df exist in avg_temps
mapp_formulas = set(mapp_df['chemical_formula'].unique())
missing_formulas_mapp = mapp_formulas - avg_temps_formulas

if missing_formulas_mapp:
    print(f"\nWARNING: {len(missing_formulas_mapp)} formulas in mapp_df are NOT in ternary_df:")
    print(f"Missing formulas: {sorted(list(missing_formulas_mapp)[:10])}...")  # Show first 10
else:
    print(f"\nAll {len(mapp_formulas)} formulas in mapp_df exist in ternary_df ✓")

# Update mpds_melting_point_kelvin in mapp_df
mapp_df['mpds_melting_point_kelvin'] = mapp_df['chemical_formula'].map(temp_mapping)

# Check for any NaN values (formulas that couldn't be mapped)
unmapped_count_mapp = mapp_df['mpds_melting_point_kelvin'].isna().sum()
if unmapped_count_mapp > 0:
    print(f"\nWARNING: {unmapped_count_mapp} rows could not be mapped to temperatures")
else:
    print(f"\nSuccessfully updated all {len(mapp_df)} rows with IQR-filtered temperatures ✓")

# Save updated mapp_df
output_file_mapp = "data/ternary_dft_data/output_updated.csv"
mapp_df.to_csv(output_file_mapp, index=False)
print(f"\nSaved updated data to: {output_file_mapp}")
