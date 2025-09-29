
"""
Script for analyzing melting temperature deviations using machine learning.

This script extracts features from metadata and uses Random Forest with SHAP analysis
to identify which attributes explain the deviation between predicted and experimental
melting temperatures.

Usage Options:
1. Run analysis on all systems (default)
2. Filter out systems with "Contains predicted" Fit Type by setting exclude_predicted=True in main()
3. Run comparison analysis by uncommenting run_comparison_analysis() call in main()

Features included: MAE/RMSE statistics, L-parameter statistics, formation energies
Features excluded: Categorical variables (Fit Type, type, congruent status)
"""

import pandas as pd
import numpy as np
import json
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SHAP, handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not available. Feature importance analysis will use Random Forest importance only.")
    print("To install SHAP: pip install shap")

inter_path = "all_dumps/gliq_manu_test3/ternary_Gliq_mps_final_linear.xlsx"
meta_data_path = "all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"


def extract_features_from_metadata():
    """
    Extract features from the metadata JSON and merge with Excel data.
    """
    # Load Excel data
    df = pd.read_excel(inter_path)
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Calculate target variable: deviation between predicted and experimental
    df['temp_deviation'] = df['gliq_melting_temp'] - df['melting_point_k']
    df['abs_temp_deviation'] = np.abs(df['temp_deviation'])
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Initialize feature lists
    features = []
    system_keys = []
    
    for _, row in df.iterrows():
        system_key = row['system_key']
        system_keys.append(system_key)
        
        if system_key in meta_data:
            system_meta = meta_data[system_key]
            feature_dict = {}
            
            # Note: Categorical features (Fit Type, type, congruent status) excluded from analysis
            # Note: mpds_temp and calculated_temp excluded as they are the same as melting_point_k and gliq_melting_temp
            
            # Error metrics (MAE and RMSE)
            mae_values = system_meta.get('mae', [])
            rmse_values = system_meta.get('rmse', [])
            norm_mae_values = system_meta.get('norm_mae', [])
            norm_rmse_values = system_meta.get('norm_rmse', [])
            
            feature_dict['mae_mean'] = np.mean(mae_values) if mae_values else 0
            feature_dict['mae_std'] = np.std(mae_values) if len(mae_values) > 1 else 0
            feature_dict['mae_max'] = np.max(mae_values) if mae_values else 0
            feature_dict['mae_min'] = np.min(mae_values) if mae_values else 0
            
            feature_dict['rmse_mean'] = np.mean(rmse_values) if rmse_values else 0
            feature_dict['rmse_std'] = np.std(rmse_values) if len(rmse_values) > 1 else 0
            feature_dict['rmse_max'] = np.max(rmse_values) if rmse_values else 0
            feature_dict['rmse_min'] = np.min(rmse_values) if rmse_values else 0
            
            feature_dict['norm_mae_mean'] = np.mean(norm_mae_values) if norm_mae_values else 0
            feature_dict['norm_rmse_mean'] = np.mean(norm_rmse_values) if norm_rmse_values else 0
            
            # Ternary hull metrics
            ternary_meta = system_meta.get('ternary_meta', {})
            feature_dict['n_ternary_compounds'] = ternary_meta.get('n_ternary_compounds', 0)
            feature_dict['deepest_formation_energy'] = ternary_meta.get('deepest_formation_energy', 0)
            feature_dict['abs_deepest_formation_energy'] = abs(ternary_meta.get('deepest_formation_energy', 0))
            
            # Binary L parameter statistics
            binary_params = system_meta.get('binary_L_params', {})
            if binary_params:
                # Collect all L parameters
                L0_a_values = []
                L0_b_values = []
                L1_a_values = []
                
                for binary_key, params in binary_params.items():
                    if len(params) >= 4:
                        L0_a_values.append(params[0])
                        L0_b_values.append(params[1])
                        L1_a_values.append(params[2])
                
                if len(L0_a_values) >= 3:  # Ensure we have all 3 binaries
                    # Mean values
                    feature_dict['L0_a_mean'] = np.mean(L0_a_values)
                    feature_dict['L0_b_mean'] = np.mean(L0_b_values)
                    feature_dict['L1_a_mean'] = np.mean(L1_a_values)
                    
                    # Standard deviations (variation across binaries)
                    feature_dict['L0_a_std'] = np.std(L0_a_values)
                    feature_dict['L0_b_std'] = np.std(L0_b_values)
                    feature_dict['L1_a_std'] = np.std(L1_a_values)
                    
                    # Ranges
                    feature_dict['L0_a_range'] = np.max(L0_a_values) - np.min(L0_a_values)
                    feature_dict['L0_b_range'] = np.max(L0_b_values) - np.min(L0_b_values)
                    feature_dict['L1_a_range'] = np.max(L1_a_values) - np.min(L1_a_values)
                    
                    # Average magnitudes
                    feature_dict['L0_a_avg_magnitude'] = np.mean(np.abs(L0_a_values))
                    feature_dict['L0_b_avg_magnitude'] = np.mean(np.abs(L0_b_values))
                    feature_dict['L1_a_avg_magnitude'] = np.mean(np.abs(L1_a_values))
                    
                    # Maximum magnitudes
                    feature_dict['L0_a_max_magnitude'] = np.max(np.abs(L0_a_values))
                    feature_dict['L0_b_max_magnitude'] = np.max(np.abs(L0_b_values))
                    feature_dict['L1_a_max_magnitude'] = np.max(np.abs(L1_a_values))
                else:
                    # Fill with zeros if incomplete data
                    for param in ['L0_a', 'L0_b', 'L1_a']:
                        for stat in ['mean', 'std', 'range', 'avg_magnitude', 'max_magnitude']:
                            feature_dict[f'{param}_{stat}'] = 0
            else:
                # Fill with zeros if no binary params
                for param in ['L0_a', 'L0_b', 'L1_a']:
                    for stat in ['mean', 'std', 'range', 'avg_magnitude', 'max_magnitude']:
                        feature_dict[f'{param}_{stat}'] = 0
            
            features.append(feature_dict)
        else:
            # Fill with zeros if system not found in metadata (excluding categorical features)
            feature_dict = {key: 0 for key in [
                'mae_mean', 'mae_std', 'mae_max', 'mae_min',
                'rmse_mean', 'rmse_std', 'rmse_max', 'rmse_min', 'norm_mae_mean', 'norm_rmse_mean',
                'n_ternary_compounds', 'deepest_formation_energy', 'abs_deepest_formation_energy'
            ]}
            for param in ['L0_a', 'L0_b', 'L1_a']:
                for stat in ['mean', 'std', 'range', 'avg_magnitude', 'max_magnitude']:
                    feature_dict[f'{param}_{stat}'] = 0
            features.append(feature_dict)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    features_df['system_key'] = system_keys
    
    # Merge with original data
    result_df = df.merge(features_df, on='system_key', how='left')
    
    return result_df


def analyze_melting_temp_deviation(exclude_contains_predicted=False):
    """
    Use machine learning to identify which metadata attributes explain 
    the deviation between gliq_melting_temp and melting_point_k.
    
    Args:
        exclude_contains_predicted (bool): If True, exclude all systems with "Contains predicted" Fit Type
    """
    print("Extracting features from metadata...")
    df = extract_features_from_metadata()
    
    print(f"Initial dataset shape: {df.shape}")
    
    # Filter out systems with "Contains predicted" if requested
    if exclude_contains_predicted:
        print("\nFiltering out systems with 'Contains predicted' Fit Type...")
        
        # Load metadata to identify systems to exclude
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)
        
        # Find system keys with "Contains predicted" Fit Type
        excluded_systems = []
        for system_key, system_meta in meta_data.items():
            if system_meta.get('Fit Type') == 'Contains predicted':
                excluded_systems.append(system_key)
        
        print(f"Found {len(excluded_systems)} systems with 'Contains predicted' Fit Type:")
        for system in excluded_systems[:10]:  # Show first 10
            print(f"  - {system}")
        if len(excluded_systems) > 10:
            print(f"  ... and {len(excluded_systems) - 10} more")
        
        # Filter the dataframe
        initial_count = len(df)
        df = df[~df['system_key'].isin(excluded_systems)]
        filtered_count = len(df)
        
        print(f"Filtered out {initial_count - filtered_count} systems")
        print(f"Remaining systems: {filtered_count}")
        
        if filtered_count == 0:
            print("ERROR: No systems remaining after filtering!")
            return
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Temperature deviation range: {df['temp_deviation'].min():.2f} to {df['temp_deviation'].max():.2f}")
    print(f"Absolute temperature deviation range: {df['abs_temp_deviation'].min():.2f} to {df['abs_temp_deviation'].max():.2f}")
    
    # Prepare features (exclude non-feature columns and categorical features)
    # Only numerical features are included: error metrics, L parameters, formation energies, etc.
    feature_columns = [col for col in df.columns if col not in [
        'reduced_formula', 'melting_point_k', 'elements', 'gliq_melting_temp', 
        'system_key', 'temp_deviation', 'abs_temp_deviation', 'type'
    ]]
    
    X = df[feature_columns].fillna(0)  # Fill any remaining NaN values
    y_deviation = df['temp_deviation']  # Signed deviation
    y_abs_deviation = df['abs_temp_deviation']  # Absolute deviation
    
    print(f"Number of features: {len(feature_columns)}")
    print("Feature categories:")
    print("- Error metrics (MAE, RMSE):", [col for col in feature_columns if any(x in col for x in ['mae', 'rmse'])])
    print("- L parameter statistics:", [col for col in feature_columns if any(x in col for x in ['L0_', 'L1_'])])
    print("- Formation energy features:", [col for col in feature_columns if 'formation' in col])
    print("- Other features:", [col for col in feature_columns if not any(x in col for x in ['mae', 'rmse', 'L0_', 'L1_', 'formation'])])
    print("\nAll features:", feature_columns)
    
    # Analysis 1: Predict signed temperature deviation
    print("\n" + "="*60)
    print("ANALYSIS 1: Predicting Signed Temperature Deviation")
    print("="*60)
    
    results_signed = analyze_target(X, y_deviation, "Signed Temperature Deviation", feature_columns)
    
    # Analysis 2: Predict absolute temperature deviation
    print("\n" + "="*60)
    print("ANALYSIS 2: Predicting Absolute Temperature Deviation")
    print("="*60)
    
    results_abs = analyze_target(X, y_abs_deviation, "Absolute Temperature Deviation", feature_columns)
    
    # Feature correlation analysis
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_analysis(df, feature_columns)
    
    # Compare feature importance methods
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*60)
    
    compare_feature_importance(results_signed, results_abs)


def analyze_target(X, y, target_name, feature_names):
    """
    Analyze a specific target variable using Random Forest.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    
    # Create Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit model
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    
    print(f"\nRandom Forest Results:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance analysis
    print(f"\nFeature Importance Analysis for {target_name}:")
    print("-" * 50)
    
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Feature Importances:")
    for idx, row in rf_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = rf_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances - {target_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # SHAP Analysis
    if SHAP_AVAILABLE:
        print(f"\nSHAP Analysis for {target_name}:")
        print("-" * 50)
        
        try:
            # Create SHAP explainer for tree-based models
            explainer = shap.TreeExplainer(rf_model)
            
            # Calculate SHAP values for test set (or subset if too large)
            sample_size = min(100, len(X_test))  # Use up to 100 samples for speed
            X_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            print(f"Calculated SHAP values for {sample_size} samples")
            
            # Summary plot (bar plot of mean absolute SHAP values)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                             plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {target_name}')
            plt.tight_layout()
            plt.show()
            
            # Detailed summary plot (beeswarm plot)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {target_name}')
            plt.tight_layout()
            plt.show()
            
            # Calculate mean absolute SHAP values for ranking
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            print("Top 10 Features by Mean Absolute SHAP Value:")
            for idx, row in shap_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
            
            # Waterfall plot for first prediction
            if len(X_sample) > 0:
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(explainer.expected_value, shap_values[0], 
                                  X_sample.iloc[0] if hasattr(X_sample, 'iloc') else X_sample[0],
                                  feature_names=feature_names, show=False)
                plt.title(f'SHAP Waterfall Plot - First Sample - {target_name}')
                plt.tight_layout()
                plt.show()
                
            return rf_model, rf_importance, shap_importance
            
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            print("Continuing with standard feature importance analysis...")
            return rf_model, rf_importance
    else:
        print(f"\nSHAP not available. Using Random Forest feature importance only.")
        return rf_model, rf_importance


def correlation_analysis(df, feature_columns):
    """
    Analyze correlations between features and target variables.
    """
    # Create correlation matrix
    analysis_columns = feature_columns + ['temp_deviation', 'abs_temp_deviation']
    corr_matrix = df[analysis_columns].corr()
    
    # Extract correlations with target variables
    temp_dev_corr = corr_matrix['temp_deviation'].drop('temp_deviation').abs().sort_values(ascending=False)
    abs_temp_dev_corr = corr_matrix['abs_temp_deviation'].drop('abs_temp_deviation').abs().sort_values(ascending=False)
    
    print("Top 10 correlations with Signed Temperature Deviation:")
    for feature, corr in temp_dev_corr.head(10).items():
        actual_corr = corr_matrix.loc[feature, 'temp_deviation']
        print(f"  {feature}: {actual_corr:.4f} (|{corr:.4f}|)")
    
    print("\nTop 10 correlations with Absolute Temperature Deviation:")
    for feature, corr in abs_temp_dev_corr.head(10).items():
        actual_corr = corr_matrix.loc[feature, 'abs_temp_deviation']
        print(f"  {feature}: {actual_corr:.4f} (|{corr:.4f}|)")
    
    # Correlation heatmap for top features
    top_features = list(set(temp_dev_corr.head(10).index.tolist() + abs_temp_dev_corr.head(10).index.tolist()))
    plot_columns = top_features + ['temp_deviation', 'abs_temp_deviation']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[plot_columns].corr(), annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Correlation Matrix - Top Features and Target Variables')
    plt.tight_layout()
    plt.show()


def compare_feature_importance(results_signed, results_abs):
    """
    Compare feature importance between standard Random Forest and SHAP methods.
    """
    if len(results_signed) >= 3 and len(results_abs) >= 3:
        # Both analyses have SHAP results
        rf_signed, shap_signed = results_signed[1], results_signed[2]
        rf_abs, shap_abs = results_abs[1], results_abs[2]
        
        print("Comparison: Random Forest vs SHAP Importance")
        print("=" * 50)
        
        # Merge and compare for signed deviation
        print("\nSigned Temperature Deviation - Top 10 Features:")
        print("RF Importance vs SHAP Importance")
        print("-" * 50)
        
        comparison_signed = rf_signed.merge(shap_signed, on='feature', suffixes=('_rf', '_shap'))
        comparison_signed = comparison_signed.sort_values('importance_rf', ascending=False).head(10)
        
        for _, row in comparison_signed.iterrows():
            print(f"{row['feature']:<30} RF: {row['importance_rf']:.4f}  SHAP: {row['mean_abs_shap']:.4f}")
        
        # Merge and compare for absolute deviation
        print("\nAbsolute Temperature Deviation - Top 10 Features:")
        print("RF Importance vs SHAP Importance")
        print("-" * 50)
        
        comparison_abs = rf_abs.merge(shap_abs, on='feature', suffixes=('_rf', '_shap'))
        comparison_abs = comparison_abs.sort_values('importance_rf', ascending=False).head(10)
        
        for _, row in comparison_abs.iterrows():
            print(f"{row['feature']:<30} RF: {row['importance_rf']:.4f}  SHAP: {row['mean_abs_shap']:.4f}")
        
        # Correlation between importance methods
        print("\nCorrelation between RF and SHAP importance:")
        corr_signed = np.corrcoef(comparison_signed['importance_rf'], comparison_signed['mean_abs_shap'])[0,1]
        corr_abs = np.corrcoef(comparison_abs['importance_rf'], comparison_abs['mean_abs_shap'])[0,1]
        print(f"Signed deviation: {corr_signed:.4f}")
        print(f"Absolute deviation: {corr_abs:.4f}")
        
        # Scatter plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Signed deviation comparison
        ax1.scatter(comparison_signed['importance_rf'], comparison_signed['mean_abs_shap'])
        ax1.set_xlabel('Random Forest Importance')
        ax1.set_ylabel('Mean Absolute SHAP Value')
        ax1.set_title(f'Signed Deviation\nCorrelation: {corr_signed:.4f}')
        
        # Add diagonal line
        min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Absolute deviation comparison
        ax2.scatter(comparison_abs['importance_rf'], comparison_abs['mean_abs_shap'])
        ax2.set_xlabel('Random Forest Importance')
        ax2.set_ylabel('Mean Absolute SHAP Value')
        ax2.set_title(f'Absolute Deviation\nCorrelation: {corr_abs:.4f}')
        
        # Add diagonal line
        min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
        max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("SHAP analysis was not completed for comparison.")
        print("Showing Random Forest feature importance only:")
        
        if len(results_signed) >= 2 and len(results_abs) >= 2:
            rf_signed = results_signed[1]
            rf_abs = results_abs[1]
            
            print("\nTop 10 Features - Signed Temperature Deviation:")
            print("-" * 50)
            for _, row in rf_signed.head(10).iterrows():
                print(f"{row['feature']:<30} RF: {row['importance']:.4f}")
            
            print("\nTop 10 Features - Absolute Temperature Deviation:")
            print("-" * 50)
            for _, row in rf_abs.head(10).iterrows():
                print(f"{row['feature']:<30} RF: {row['importance']:.4f}")


def run_comparison_analysis():
    """
    Run both analyses for comparison: with and without "Contains predicted" systems.
    """
    print("="*80)
    print("COMPARISON ANALYSIS: ALL SYSTEMS vs FITTED SYSTEMS ONLY")
    print("="*80)
    
    print("\n" + "="*60)
    print("ANALYSIS 1: ALL SYSTEMS (including 'Contains predicted')")
    print("="*60)
    analyze_melting_temp_deviation(exclude_contains_predicted=False)
    
    print("\n" + "="*60)
    print("ANALYSIS 2: FITTED SYSTEMS ONLY (excluding 'Contains predicted')")
    print("="*60)
    analyze_melting_temp_deviation(exclude_contains_predicted=True)


def main():
    """
    Main function to run the analysis.
    """
    print("Starting melting temperature deviation analysis...")
    print("\nAnalysis Options:")
    print("1. Include all systems")
    print("2. Exclude systems with 'Contains predicted' Fit Type")
    
    # You can change this to True to exclude "Contains predicted" systems
    exclude_predicted = False  # Set to True to filter out "Contains predicted" systems
    
    # Uncomment the line below to run both analyses for comparison
    # run_comparison_analysis(); return
    
    if exclude_predicted:
        print("\n>>> Running analysis EXCLUDING 'Contains predicted' systems <<<")
    else:
        print("\n>>> Running analysis INCLUDING all systems <<<")
    
    analyze_melting_temp_deviation(exclude_contains_predicted=exclude_predicted)


if __name__ == "__main__":
    main()