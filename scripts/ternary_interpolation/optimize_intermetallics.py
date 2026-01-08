from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
# from scipy.optimize import minimize_scalar, brentq

dump_dir = "all_dumps/gliq_manu_test7_linear/"
meta_doc = dump_dir + "ternary_Gliq_meta_final_linear.json"
final_dir = "all_dumps/gliq_manu_test7_correction3/"
read_dir = "all_dumps/binary_fits/"
print(data_dir)

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

if not os.path.exists(final_dir):
    os.makedirs(final_dir)


# Set to a specific formula to test a single system, or None to process all
# For optimization: 
# TEST_SINGLE_FORMULA = "Tm(CuGe)2"
TEST_SINGLE_FORMULA = None  # Uncomment this line to process all systems

# TEST_SINGLE_SYSTEM = ["In", "Ag", "Se"]
TEST_SINGLE_SYSTEM = None 
# ============================================================================


def optimize_l0_tern(tern_sys, binary_L_dict, fitorpred, tern_param_format, interp, 
                      congruent_phase, target_temp, initial_delta_T, max_iterations=10):
    """
    Optimize l0_tern to minimize temperature error using adaptive iterative approach.
    
    Args:
        tern_sys: List of elements in ternary system
        binary_L_dict: Dictionary of binary L parameters
        fitorpred: Dictionary indicating fit or pred for each binary
        tern_param_format: Parameter format (e.g., 'combined')
        interp: Interpolation type (e.g., 'linear')
        congruent_phase: Phase formula to optimize
        target_temp: Target melting temperature in K
        initial_delta_T: Initial temperature error (predicted - actual)
        max_iterations: Maximum optimization iterations
    
    Returns:
        tuple: (optimal_l0_tern, final_predicted_temp, success_flag, error_message)
    """
    
    # Check if optimization is needed
    acceptable_error = 10.0  # +/- 10K
    
    if abs(initial_delta_T) <= acceptable_error:
        print(f"\n  Initial error ({initial_delta_T:+.1f}K) already within acceptable range. Skipping optimization.")
        return 0.0, target_temp + initial_delta_T, True, "Success - no optimization needed", None
    
    def evaluate_temp(l0_tern_value):
        """Evaluate predicted temperature for a given l0_tern value."""
        error_details = {}
        try:
            print(f"    Testing l0_tern = {l0_tern_value:.0f} J/mol...", end=" ")
            
            plotter = ternary_gtx_plotter(
                tern_sys, data_dir, 
                interp_type=interp, 
                param_format=tern_param_format,
                L_dict=binary_L_dict, 
                temp_slider=[500, 500], 
                T_incr=5, 
                delta=0.025, 
                fit_or_pred=fitorpred,
                L_tern=[l0_tern_value, 0]
            )
            
            error_details['stage'] = 'interpolation'
            plotter.interpolate()
            
            error_details['stage'] = 'process_data'
            plotter.process_data()

            error_details['stage'] = 'plot_ternary'
            tern_fig = plotter.plot_ternary()

            # Check minimum liquidus temperature constraint
            error_details['stage'] = 'temperature_validation'
            if hasattr(plotter, 'liq_plotting_df') and plotter.liq_plotting_df is not None:
                if 'T' in plotter.liq_plotting_df.columns:
                    min_temp_celsius = plotter.liq_plotting_df['T'].min()
                    if min_temp_celsius < -270:
                        error_msg = f"Minimum liquidus temperature ({min_temp_celsius:.1f}°C) below -270°C threshold"
                        print(f"TEMP CONSTRAINT VIOLATION: {min_temp_celsius:.1f}°C")
                        return None, error_msg
            
            error_details['stage'] = 'get_melting_temps'
            inter_list = [congruent_phase]
            melting_temps = plotter.get_inter_melting_temps(inter_list)
            
            if melting_temps and congruent_phase in melting_temps:
                # get_inter_melting_temps returns temperature in CELSIUS
                # Convert to Kelvin for comparison with target_temp
                temp_result_celsius = melting_temps[congruent_phase]
                temp_result_kelvin = temp_result_celsius + 273.15
                print(f"T = {temp_result_kelvin:.1f}K (error: {temp_result_kelvin - target_temp:+.1f}K)")
                return temp_result_kelvin, None
            else:
                error_msg = f"Phase '{congruent_phase}' not found in results. Available: {list(melting_temps.keys()) if melting_temps else 'None'}"
                print(f"PHASE NOT FOUND")
                return None, error_msg
                
        except Exception as e:
            error_msg = f"Error at {error_details.get('stage', 'unknown stage')}: {str(e)}"
            print(f"ERROR: {str(e)[:50]}...")
            return None, error_msg
    
    print(f"\n  Starting iterative optimization...")
    print(f"  Initial error: {initial_delta_T:+.1f}K (target ≤ ±{acceptable_error:.0f}K)")
    
    # Adaptive iterative optimization
    current_l0 = 0.0
    current_delta_T = initial_delta_T
    iteration = 0
    error_log = []
    
    best_l0 = 0.0
    best_temp = target_temp + initial_delta_T
    best_error = abs(initial_delta_T)
    
    failure_stages = []  # Track stages where failures occurred
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n  Iteration {iteration}/{max_iterations}:")
        
        # Calculate correction based on current error
        # delta_T = predicted - actual
        # If delta_T > 0 (predicted too high), need negative l0_tern
        # If delta_T < 0 (predicted too low), need positive l0_tern
        correction_ratio = -800  # J/mol per K
        # correction_ratio = -500  # J/mol per K
        l0_correction = correction_ratio * current_delta_T
        
        # Apply correction
        test_l0 = current_l0 + l0_correction
        
        print(f"    Current l0_tern: {current_l0:.0f} J/mol")
        print(f"    Current error: {current_delta_T:+.1f}K")
        print(f"    Applying correction: {l0_correction:+.0f} J/mol")
        
        # Evaluate new l0_tern value
        temp_result, error_msg = evaluate_temp(test_l0)
        
        if temp_result is None:
            error_log.append((test_l0, error_msg))
            # Extract failure stage from error message
            if 'Error at' in error_msg:
                stage = error_msg.split('Error at ')[1].split(':')[0]
                failure_stages.append(stage)
            
            # Check if failure is due to temperature constraint violation
            if error_msg and 'below -270°C threshold' in error_msg:
                print(f"    Temperature constraint violated - reducing l0_tern magnitude...")
                
                # Reduce magnitude iteratively until constraint is satisfied
                for scale_factor in [0.75, 0.5, 0.25, 0.1]:
                    test_l0_reduced = test_l0 * scale_factor
                    print(f"    Trying {scale_factor*100:.0f}% of l0_tern: {test_l0_reduced:.0f} J/mol")
                    
                    temp_result, error_msg = evaluate_temp(test_l0_reduced)
                    if temp_result is not None:
                        test_l0 = test_l0_reduced
                        print(f"    Temperature constraint satisfied at {scale_factor*100:.0f}% magnitude")
                        break
                    else:
                        error_log.append((test_l0_reduced, error_msg))
                        if error_msg and 'below -270°C threshold' not in error_msg:
                            # Different error, stop trying
                            if 'Error at' in error_msg:
                                stage = error_msg.split('Error at ')[1].split(':')[0]
                                failure_stages.append(stage)
                            break
            else:
                print(f"    Failed to evaluate - trying reduced correction...")
                
                # Try smaller correction if full correction failed
                for scale_factor in [0.5, 0.25]:
                    reduced_correction = l0_correction * scale_factor
                    test_l0_reduced = current_l0 + reduced_correction
                    print(f"    Trying {scale_factor*100:.0f}% correction: {reduced_correction:+.0f} J/mol")
                    
                    temp_result, error_msg = evaluate_temp(test_l0_reduced)
                    if temp_result is not None:
                        test_l0 = test_l0_reduced
                        break
                    else:
                        error_log.append((test_l0_reduced, error_msg))
                        if 'Error at' in error_msg:
                            stage = error_msg.split('Error at ')[1].split(':')[0]
                            failure_stages.append(stage)
            
            if temp_result is None:
                print(f"    All attempts failed in this iteration")
                break
        
        # Update tracking
        new_delta_T = temp_result - target_temp
        new_error = abs(new_delta_T)
        
        if new_error < best_error:
            best_error = new_error
            best_l0 = test_l0
            best_temp = temp_result
        
        print(f"    New error: {new_delta_T:+.1f}K (|{new_error:.1f}K|)")
        
        # Check if acceptable
        if new_error <= acceptable_error:
            print(f"\n  Converged! Final error: {new_delta_T:+.1f}K")
            return best_l0, best_temp, True, "Success", None
        
        # Check if we're making progress
        if abs(new_delta_T) >= abs(current_delta_T) * 0.95:
            # Not making sufficient progress
            print(f"    Progress stalled (error reduction < 5%)")
            break
        
        # Update for next iteration
        current_l0 = test_l0
        current_delta_T = new_delta_T
    
    # Did not converge within iterations
    most_common_failure = None
    if failure_stages:
        from collections import Counter
        failure_counts = Counter(failure_stages)
        most_common_failure = failure_counts.most_common(1)[0][0]
    
    if best_error < abs(initial_delta_T):
        error_message = f"Partial success (error: {best_error:.1f}K, improved from {abs(initial_delta_T):.1f}K)"
        print(f"\n  Partial success - best error: {best_error:.1f}K")
    else:
        error_message = f"Failed - no improvement (error: {best_error:.1f}K)"
        print(f"\n  Failed - no improvement achieved")
    
    if error_log:
        print(f"\n  Errors encountered: {len(error_log)}")
        if most_common_failure:
            print(f"  Most common failure stage: {most_common_failure}")
    
    return best_l0, best_temp, False, error_message, most_common_failure


def main_optimize():
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"

    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")
    ternary_df = pd.read_excel(dump_dir + "ternary_Gliq_mps_final_linear_updated.xlsx")
    
    # Load metadata to filter for "All fitted" systems only
    if not os.path.exists(meta_doc):
        print(f"WARNING: Metadata file not found at {meta_doc}")
        print("Proceeding without metadata filtering...")
        metadata = {}
    else:
        with open(meta_doc, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from: {meta_doc}")
    
    # Filter for single system testing if specified
    if TEST_SINGLE_FORMULA is not None:
        ternary_df = ternary_df[ternary_df['reduced_formula'] == TEST_SINGLE_FORMULA]
        if len(ternary_df) == 0:
            print(f"ERROR: Formula '{TEST_SINGLE_FORMULA}' not found in the dataset!")
            print(f"Available formulas: {list(ternary_df['reduced_formula'].unique())[:10]}...")
            return
        print(f"\n*** TESTING MODE: Processing only {TEST_SINGLE_FORMULA} ***\n")
    
    # Prepare results storage
    results_list = []
    skipped_count = 0
    
    for idx, row in ternary_df.iterrows():
        tern_sys = ast.literal_eval(row["elements"]) if isinstance(row["elements"], str) else row["elements"]
        congruent_phase = row["reduced_formula"]
        actual_temp = row["melting_point_k"]
        initial_gliq_temp = row["gliq_melting_temp"]
        initial_delta_T = initial_gliq_temp - actual_temp
        
        # Check if this system is "All fitted" in metadata
        sorted_sys = sorted(tern_sys)
        system_key = "-".join(sorted_sys)
        
        if metadata:
            if system_key in metadata:
                fit_type = metadata[system_key].get("Fit Type", "")
                if fit_type != "All fitted":
                    print(f"\nSkipping {congruent_phase} (system: {system_key}) - Fit Type: {fit_type}")
                    skipped_count += 1
                    continue
            else:
                print(f"\nWARNING: System {system_key} not found in metadata, skipping...")
                skipped_count += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"Processing {congruent_phase} (system: {tern_sys})")
        print(f"Target: {actual_temp:.1f}K, Initial predicted: {initial_gliq_temp:.1f}K")
        print(f"Initial error: {initial_delta_T:+.1f}K")
        print(f"{'='*70}")
        
        try:
            binary_sys_labels = [
                f"{sorted_sys[0]}-{sorted_sys[1]}",
                f"{sorted_sys[1]}-{sorted_sys[2]}",
                f"{sorted_sys[2]}-{sorted_sys[0]}"
            ]
            print(binary_sys_labels)

            binary_L_dict = {}
            fitorpred = {}

            pred_tag = "All fitted"
            mae = []
            rmse = []
            norm_mae = []
            norm_rmse = []
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                    mae.append(params["mae"])
                    rmse.append(params["rmse"])
                    norm_mae.append(params["norm_mae"])
                    norm_rmse.append(params["norm_rmse"])
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                    mae.append(params["mae"])
                    rmse.append(params["rmse"])
                    norm_mae.append(params["norm_mae"])
                    norm_rmse.append(params["norm_rmse"])
                elif bin_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                    pred_tag = "Contains predicted"
                elif flipped_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                    pred_tag = "Contains predicted"
                else:
                    raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

                # Extract parameters and flip L1 signs if order was changed
                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])
                
                if order_changed:
                    # Flip L1 parameter signs when element order is reversed
                    L1_a = -L1_a
                    L1_b = -L1_b
                
                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            # Run optimization
            optimal_l0, final_temp, success, error_msg, failure_stage = optimize_l0_tern(
                tern_sys=tern_sys,
                binary_L_dict=binary_L_dict,
                fitorpred=fitorpred,
                tern_param_format=tern_param_format,
                interp=interp,
                congruent_phase=congruent_phase,
                target_temp=actual_temp,
                initial_delta_T=initial_delta_T
            )
            
            # Store results
            result = {
                'reduced_formula': congruent_phase,
                'elements': str(tern_sys),
                'melting_point_k': actual_temp,
                'initial_gliq_temp': initial_gliq_temp,
                'final_gliq_temp': final_temp if final_temp is not None else np.nan,
                'l0_tern': optimal_l0,
                'initial_error_k': initial_delta_T,
                'final_error_k': (final_temp - actual_temp) if final_temp is not None else np.nan,
                'optimization_status': error_msg,
                'failure_stage': failure_stage if failure_stage else ''
            }
            results_list.append(result)
            
            print(f"\nOptimization Results:")
            print(f"  Optimal l0_tern: {optimal_l0:.0f} J/mol")
            print(f"  Final predicted temp: {final_temp:.1f}K" if final_temp else "  Failed to obtain temperature")
            print(f"  Final error: {result['final_error_k']:+.1f}K" if final_temp else "  N/A")
            print(f"  Status: {error_msg}")
            if failure_stage:
                print(f"  Failure stage: {failure_stage}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result = {
                'reduced_formula': congruent_phase,
                'elements': str(tern_sys),
                'melting_point_k': actual_temp,
                'initial_gliq_temp': initial_gliq_temp,
                'final_gliq_temp': np.nan,
                'l0_tern': 0.0,
                'initial_error_k': initial_delta_T,
                'final_error_k': np.nan,
                'optimization_status': f"Error: {str(e)}",
                'failure_stage': 'main_loop_exception'
            }
            results_list.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results
    output_file = os.path.join(final_dir, "optimized_l0_tern_results.xlsx")
    results_df.to_excel(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Print summary statistics
    successful = results_df[results_df['optimization_status'] == 'Success']
    partial = results_df[results_df['optimization_status'].str.contains('Partial', na=False)]
    failed = results_df[results_df['optimization_status'].str.contains('Failed|Error', na=False)]
    
    print(f"\nSummary:")
    print(f"  Total systems processed: {len(results_df)}")
    print(f"  Skipped (not 'All fitted'): {skipped_count}")
    print(f"  Successful (≤10K error): {len(successful)}")
    print(f"  Partial success (>10K error): {len(partial)}")
    print(f"  Failed: {len(failed)}")
    
    if len(successful) > 0:
        print(f"\nSuccessful optimizations:")
        print(f"  Mean final error: {successful['final_error_k'].abs().mean():.2f}K")
        print(f"  Max final error: {successful['final_error_k'].abs().max():.2f}K")


def main_post():
    """
    Generate and save corrected ternary phase diagrams using optimized l0_tern values.
    Uses TEST_SINGLE_SYSTEM to optionally filter to a single chemical system for testing.
    """
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"

    # Load optimization results
    results_file = os.path.join(final_dir, "optimized_l0_tern_results.xlsx")
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found at {results_file}")
        print("Please run main_optimize() first.")
        return
    
    results_df = pd.read_excel(results_file)
    
    # Filter for single system testing if specified
    if TEST_SINGLE_SYSTEM is not None:
        # Normalize the test system (sort elements)
        test_sys_sorted = sorted(TEST_SINGLE_SYSTEM)
        
        # Filter rows where elements match the test system
        def match_system(elements_str):
            elements = ast.literal_eval(elements_str) if isinstance(elements_str, str) else elements_str
            return sorted(elements) == test_sys_sorted
        
        results_df = results_df[results_df['elements'].apply(match_system)]
        
        if len(results_df) == 0:
            print(f"ERROR: System {TEST_SINGLE_SYSTEM} not found in the results!")
            return
        print(f"\n*** TESTING MODE: Plotting only system {test_sys_sorted} ***")
        print(f"Found {len(results_df)} phase(s) in this system\n")
    
    print(f"\n{'='*70}")
    print(f"Generating corrected phase diagrams from: {results_file}")
    print(f"{'='*70}")
    print(f"Total systems to plot: {len(results_df)}")
    
    # Load binary parameter data
    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")
    
    success_count = 0
    error_count = 0
    
    for idx, row in results_df.iterrows():
        tern_sys = ast.literal_eval(row["elements"]) if isinstance(row["elements"], str) else row["elements"]
        congruent_phase = row["reduced_formula"]
        optimized_l0 = row["l0_tern"]
        
        print(f"\n[{idx+1}/{len(results_df)}] Processing {congruent_phase} (system: {tern_sys})")
        print(f"  Using l0_tern = {optimized_l0:.0f} J/mol")
        
        try:
            sorted_sys = sorted(tern_sys)
            binary_sys_labels = [
                f"{sorted_sys[0]}-{sorted_sys[1]}",
                f"{sorted_sys[1]}-{sorted_sys[2]}",
                f"{sorted_sys[2]}-{sorted_sys[0]}"
            ]
            
            binary_L_dict = {}
            fitorpred = {}
            
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif bin_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                elif flipped_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                else:
                    raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

                # Extract parameters and flip L1 signs if order was changed
                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])
                
                if order_changed:
                    L1_a = -L1_a
                    L1_b = -L1_b
                
                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            # Generate ternary phase diagram with optimized l0_tern
            plotter = ternary_gtx_plotter(
                tern_sys, data_dir, 
                interp_type=interp, 
                param_format=tern_param_format,
                L_dict=binary_L_dict, 
                temp_slider=[500, 500], 
                T_incr=5, 
                delta=0.025, 
                fit_or_pred=fitorpred,
                L_tern=[optimized_l0, 0]
            )
            
            plotter.interpolate()
            plotter.process_data()
            tern_fig = plotter.plot_ternary()
            
            # Save figure to correction directory
            output_filename = f'{"-".join(sorted_sys)}_{congruent_phase}_corrected.html'
            output_path = os.path.join(final_dir, output_filename)
            ploff.plot(tern_fig, filename=output_path, auto_open=False)
            
            print(f"  Saved: {output_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"Phase diagram generation complete")
    print(f"{'='*70}")
    print(f"  Successfully plotted: {success_count}/{len(results_df)}")
    print(f"  Errors: {error_count}/{len(results_df)}")
    print(f"  Saved to: {final_dir}")


if __name__ == "__main__":
    # main_optimize()
    main_post()
