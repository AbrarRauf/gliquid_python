from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import DBSCAN
import pprint

work_dir = "all_dumps/ternary_test/"

# set to easily identify files associated with different test runs
test_prefix = "A"



def check_liquidus_continuity(liq_df, delta, gap_threshold=0.1):
    # Filter for liquid phase only
    liquid_df = liq_df[liq_df['Phase'] == 'L'].copy()
    
    if len(liquid_df) == 0:
        return {
            'is_continuous': False,
            'missing_points': [],
            'gap_regions': [{'description': 'No liquid phase data found'}],
            'coverage_fraction': 0.0
        }
    
    # Generate expected grid points for ternary system
    grid_points = []
    x0_vals = np.arange(0, 1 + delta/2, delta)
    
    for x0 in x0_vals:
        x1_vals = np.arange(0, 1 - x0 + delta/2, delta)
        for x1 in x1_vals:
            x2 = 1 - x0 - x1
            if x2 >= -1e-10:  
                grid_points.append((round(x0, 6), round(x1, 6)))
    
    grid_points = list(set(grid_points))  
    total_grid_points = len(grid_points)
    
    # Get actual liquid phase points using original composition coordinates
    liquid_points = set()
    for _, row in liquid_df.iterrows():
        x0_round = round(row['x0_orig'] / delta) * delta
        x1_round = round(row['x1_orig'] / delta) * delta
        liquid_points.add((round(x0_round, 6), round(x1_round, 6)))
    
    # Find missing points
    missing_points = [pt for pt in grid_points if pt not in liquid_points]
    coverage_fraction = len(liquid_points) / total_grid_points if total_grid_points > 0 else 0
    
    # Cluster missing points to find gap regions
    gap_regions = []
    
    if len(missing_points) > 0:
        # Use DBSCAN to cluster missing points
        # We use gap_threshold as the clustering distance
        missing_array = np.array(missing_points)
        
        clustering = DBSCAN(eps=gap_threshold/2, min_samples=1).fit(missing_array)
        labels = clustering.labels_
        
        unique_labels = set(labels)

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = missing_array[cluster_mask]
            cluster_size = len(cluster_points)
            
            # Calculate cluster extent 
            if cluster_size > 0:
                x0_min, x1_min = cluster_points.min(axis=0)
                x0_max, x1_max = cluster_points.max(axis=0)
                x0_extent = x0_max - x0_min
                x1_extent = x1_max - x1_min
                max_extent = max(x0_extent, x1_extent)
                
                # Only report gaps larger than threshold
                if max_extent >= gap_threshold:
                    gap_regions.append({
                        'num_missing_points': cluster_size,
                        'x0_range': (float(x0_min), float(x0_max)),
                        'x1_range': (float(x1_min), float(x1_max)),
                        'max_extent': float(max_extent),
                        'centroid': (float(cluster_points[:, 0].mean()), 
                                   float(cluster_points[:, 1].mean()))
                    })
    
    # Determine if surface is continuous
    is_continuous = len(gap_regions) == 0
    
    result = {
        'is_continuous': is_continuous,
        'missing_points': missing_points,
        'gap_regions': gap_regions,
        'coverage_fraction': coverage_fraction,
        'total_grid_points': total_grid_points,
        'liquid_points_count': len(liquid_points),
        'missing_points_count': len(missing_points)
    }
    
    return result


def format_continuity_for_results(continuity_result):
    """Format continuity results for storage in results dictionary."""
    formatted = {
        'is_continuous': continuity_result['is_continuous'],
        'coverage_fraction': continuity_result['coverage_fraction'],
        'total_grid_points': continuity_result['total_grid_points'],
        'liquid_points_count': continuity_result['liquid_points_count'],
        'missing_points_count': continuity_result['missing_points_count'],
        'num_gaps': len(continuity_result['gap_regions'])
    }
    
    if continuity_result['gap_regions']:
        for i, gap in enumerate(continuity_result['gap_regions']):
            formatted[f'gap{i}'] = {
                'num_missing_points': gap['num_missing_points'],
                'x0_range': gap['x0_range'],
                'x1_range': gap['x1_range'],
                'max_extent': gap['max_extent'],
                'centroid': gap['centroid']
            }
    else:
        formatted['gap_info'] = 'No significant gaps detected'
    
    return formatted


def get_eutectic_point(liquid_df, tolerance=1e-6):
    if len(liquid_df) == 0:
        return None
    
    # Calculate x2 (third component)
    liquid_df = liquid_df.copy()
    
    # Find the minimum temperature point (eutectic)
    eutectic_idx = liquid_df['T'].idxmin()
    eutectic_row = liquid_df.loc[eutectic_idx]
    
    # Ensure we have a single row (not a Series of Series)
    if isinstance(eutectic_row, pd.DataFrame):
        eutectic_row = eutectic_row.iloc[0]
    
    return {
        'temperature_K': float(eutectic_row['T']) + 273.15,
        'x0': float(eutectic_row['x0_orig']),
        'x1': float(eutectic_row['x1_orig']),
    }


def main():
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"


    '''
    ||||||||||||||||||||||||||||||||||||||||
    |||  SPECIFY TEST L PARAMETERS HERE  |||
    ||||||||||||||||||||||||||||||||||||||||
    '''

    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")

    ternary_df = work_dir + "test_set.xlsx"
    assert os.path.exists(ternary_df), f"Ternary test file {ternary_df} does not exist in working directory."

    ternary_df = pd.read_excel(ternary_df)
    ternary_sys_list = ternary_df["test_system"].tolist()
    ternary_sys_list = [ast.literal_eval(e) if isinstance(e, str) else e for e in ternary_sys_list]

    results = {}

    for i, tern_sys in enumerate(ternary_sys_list):
        sys_type = ternary_df.iloc[i]["system_type"]

        if sys_type == "intermetallic":
            inter_phase = str(ternary_df.iloc[i]["intermetallic_phase"])
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
                    raise ValueError(f"Binary system {bin_sys} not found in parameter dataframes.")

                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])

                if order_changed:
                    # Flip L1 parameter signs when element order is reversed
                    L1_a = -L1_a
                    L1_b = -L1_b

                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            # Generate ternary phase diagram with optimized l0_tern
            plotter = ternary_gtx_plotter(
                tern_sys, data_dir, 
                interp_type=interp, 
                param_format=tern_param_format,
                L_dict=binary_L_dict, 
                temp_slider=[200, 300], 
                T_incr=5, 
                delta=0.025, 
                fit_or_pred=fitorpred,
            )

            plotter.interpolate()
            plotter.process_data()
            tern_fig = plotter.plot_ternary()

            ploff.plot(tern_fig, filename=work_dir + f"{test_prefix}_ternary_test_{'_'.join(tern_sys)}.html", auto_open=False)

            melting_temps = plotter.get_inter_melting_temps([inter_phase])
            melting_temp = melting_temps[inter_phase] + 273.15
        
            delta_T = melting_temp - ternary_df.iloc[i]["melting_temp_K"]

            # Check liquidus surface continuity
            continuity_result = check_liquidus_continuity(
                plotter.liq_plotting_df, 
                delta=0.025,  # Same delta as used in plotter
                gap_threshold=0.1  # Significant gap threshold
            )
            
            # Format continuity data for results
            continuity_data = format_continuity_for_results(continuity_result)
            
            results["_".join(tern_sys)] = {
                "sys_type": sys_type,
                "predicted_melting_temp_K": melting_temp,
                "experimental_melting_temp_K": ternary_df.iloc[i]["melting_temp_K"],
                "delta_T_K": delta_T,
                "liquidus_continuity": continuity_data
            }


        else: # eutectic system 
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
                    raise ValueError(f"Binary system {bin_sys} not found in parameter dataframes.")

                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])

                if order_changed:
                    # Flip L1 parameter signs when element order is reversed
                    L1_a = -L1_a
                    L1_b = -L1_b

                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            # Generate ternary phase diagram with optimized l0_tern
            plotter = ternary_gtx_plotter(
                tern_sys, data_dir, 
                interp_type=interp, 
                param_format=tern_param_format,
                L_dict=binary_L_dict, 
                temp_slider=[200, 50], 
                T_incr=5, 
                delta=0.025, 
                fit_or_pred=fitorpred,
            )

            plotter.interpolate()
            plotter.process_data()
            tern_fig = plotter.plot_ternary()

            ploff.plot(tern_fig, filename=work_dir + f"{test_prefix}_ternary_test_{'_'.join(tern_sys)}.html", auto_open=False)
            
            # Extract eutectic point
            eutectic_point = get_eutectic_point(plotter.liq_plotting_df)
            
            # Check liquidus surface continuity
            continuity_result = check_liquidus_continuity(
                plotter.liq_plotting_df, 
                delta=0.025,  # Same delta as used in plotter
                gap_threshold=0.1  # adjust to desired significant gap threshold
            )
            
            # Format continuity data for results
            continuity_data = format_continuity_for_results(continuity_result)

            delta_T = eutectic_point['temperature_K'] - ternary_df.iloc[i]["eutectic_temp_K"]

            # find cartesian distance between predicted and experimental eutectic compositions
            eutectic_comp = ternary_df.iloc[i]["eutectic_comp"]
            if isinstance(eutectic_comp, str):
                eutectic_comp = ast.literal_eval(eutectic_comp)
            
            exp_x0, exp_x1 = eutectic_comp[0], eutectic_comp[1]
            
            # Calculate distance using only x0 and x1 (x2 is dependent)
            exp_x = np.array([exp_x0, exp_x1])
            pred_x = np.array([eutectic_point['x0'], eutectic_point['x1']])

            comp_distance = np.linalg.norm(exp_x - pred_x)
            
            results["_".join(tern_sys)] = {
                "sys_type": sys_type,
                "predicted_eutectic_temp_K": eutectic_point['temperature_K'],
                "experimental_eutectic_temp_K": ternary_df.iloc[i]["eutectic_temp_K"],
                "delta_T_K": delta_T,
                "predicted_eutectic_composition": {
                    "x0": eutectic_point['x0'],
                    "x1": eutectic_point['x1'],
                },
                "experimental_eutectic_composition": {
                    "x0": exp_x0,
                    "x1": exp_x1,
                },
                "composition_cartesian_distance": comp_distance,
                "liquidus_continuity": continuity_data
            }

    pprint.pprint(results, sort_dicts=False)

    # dump results to json file
    results_file = work_dir + f"{test_prefix}_ternary_test_report.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

                


if __name__ == "__main__":
    main()