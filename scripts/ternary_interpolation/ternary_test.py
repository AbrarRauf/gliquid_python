from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
import plotly.graph_objects as go
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
import pprint



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


def compute_liquidus_temperature_checks(liq_df, full_plotting_df=None, composition_tol=1e-6):
    liquid_df = liq_df[liq_df['Phase'] == 'L'].copy()

    if liquid_df.empty:
        return {
            'max_ternary_interior_liquidus_K': np.nan,
            'max_binary_reference_temp_K': np.nan,
            'binary_edge_maxima_K': {'x2_edge': np.nan, 'x0_edge': np.nan, 'x1_edge': np.nan},
            'ternary_binary_max_deviation_K': np.nan,
            'notes': ['No liquid phase points available for temperature diagnostics.']
        }

    liquid_df = liquid_df.copy()
    liquid_df['x2_orig'] = 1.0 - liquid_df['x0_orig'] - liquid_df['x1_orig']
    liquid_df['T_K'] = liquid_df['T'] + 273.15

    notes = []

    interior_mask = (
        (liquid_df['x0_orig'] > composition_tol)
        & (liquid_df['x1_orig'] > composition_tol)
        & (liquid_df['x2_orig'] > composition_tol)
    )
    interior_points = liquid_df[interior_mask]
    max_ternary_interior = float(interior_points['T_K'].max()) if not interior_points.empty else np.nan

    if interior_points.empty:
        notes.append('No strict ternary interior points found for max interior liquidus.')

    edge_masks = {
        'x2_edge': np.isclose(liquid_df['x2_orig'], 0.0, atol=composition_tol),
        'x0_edge': np.isclose(liquid_df['x0_orig'], 0.0, atol=composition_tol),
        'x1_edge': np.isclose(liquid_df['x1_orig'], 0.0, atol=composition_tol),
    }

    edge_maxima = {}
    for edge_name, mask in edge_masks.items():
        edge_points = liquid_df[mask]
        edge_maxima[edge_name] = float(edge_points['T_K'].max()) if not edge_points.empty else np.nan
        if edge_points.empty:
            notes.append(f'No liquid points found on binary edge {edge_name}.')

    max_binary_liquid_edge = np.nan
    valid_edge_maxima = [v for v in edge_maxima.values() if not np.isnan(v)]
    if valid_edge_maxima:
        max_binary_liquid_edge = float(np.max(valid_edge_maxima))

    max_reference_phase_temp = np.nan
    if full_plotting_df is not None and not full_plotting_df.empty:
        ref_df = full_plotting_df.copy()
        ref_df['x2_orig'] = 1.0 - ref_df['x0_orig'] - ref_df['x1_orig']
        ref_df['T_K'] = ref_df['T'] + 273.15
        pure_component_mask = (
            (ref_df['x0_orig'] >= 1.0 - composition_tol)
            | (ref_df['x1_orig'] >= 1.0 - composition_tol)
            | (ref_df['x2_orig'] >= 1.0 - composition_tol)
        )
        pure_rows = ref_df[pure_component_mask]
        if not pure_rows.empty:
            max_reference_phase_temp = float(pure_rows['T_K'].max())

    max_binary_reference_temp = np.nanmax([max_binary_liquid_edge, max_reference_phase_temp])
    if np.isnan(max_binary_reference_temp):
        max_binary_reference_temp = np.nan

    if np.isnan(max_binary_liquid_edge):
        notes.append('Could not compute max binary liquidus temperature from edge points.')

    if np.isnan(max_reference_phase_temp):
        notes.append('No pure-component reference phase temperatures found in plotting data.')

    if np.isnan(max_ternary_interior) or np.isnan(max_binary_reference_temp):
        ternary_binary_deviation = np.nan
    else:
        ternary_binary_deviation = float(max_ternary_interior - max_binary_reference_temp)

    return {
        'max_ternary_interior_liquidus_K': max_ternary_interior,
        'max_binary_reference_temp_K': max_binary_reference_temp,
        'max_reference_phase_temp_K': max_reference_phase_temp,
        'binary_edge_maxima_K': edge_maxima,
        'ternary_binary_max_deviation_K': ternary_binary_deviation,
        'notes': notes
    }


def add_liquid_scatter_trace(fig, plotter):
    if not hasattr(plotter, 'liq_plotting_df') or plotter.liq_plotting_df is None:
        return

    liq_points = plotter.liq_plotting_df.copy()
    if liq_points.empty:
        return

    liq_points = liq_points.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='first')
    fig.add_trace(go.Scatter3d(
        x=liq_points['x0'],
        y=liq_points['x1'],
        z=liq_points['T'],
        mode='markers',
        marker=dict(size=3.0, color='black', opacity=0.65),
        name='Liquid Grid Points',
        showlegend=True,
        hovertemplate=(
            '<b>Liquid Grid Point</b><br>'
            f'x_{plotter.tern_sys[1]}: %{{customdata[0]:.3f}}<br>'
            f'x_{plotter.tern_sys[2]}: %{{customdata[1]:.3f}}<br>'
            'T: %{z:.1f}°C<br>'
            '<extra></extra>'
        ),
        customdata=np.column_stack((liq_points['x0_orig'], liq_points['x1_orig'])),
    ))


def save_intermetallic_parity_plot(inter_rows, output_path):
    if not inter_rows:
        return

    exp_vals = np.array([row['experimental_K'] for row in inter_rows], dtype=float)
    pred_vals = np.array([row['predicted_K'] for row in inter_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(exp_vals, pred_vals, s=60, alpha=0.85, color='tab:blue', edgecolors='black', linewidths=0.4)

    lo = float(min(exp_vals.min(), pred_vals.min()))
    hi = float(max(exp_vals.max(), pred_vals.max()))
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=1.2)

    for row in inter_rows:
        ax.text(row['experimental_K'] + 7, row['predicted_K'] + 7, row['system'], fontsize=7, alpha=0.9)

    ax.set_xlabel('Experimental Intermetallic Temperature (K)')
    ax.set_ylabel('Predicted Intermetallic Temperature (K)')
    ax.set_title('Intermetallic Temperature Parity Plot')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def to_ternary_from_x0_x1(x0, x1):
    return [1.0 - x0 - x1, x0, x1]


def from_ternary_to_cartesian(abc):
    a, b, c = abc
    denom = a + b + c
    x = 0.5 * (2 * b + c) / denom
    y = (np.sqrt(3) / 2) * c / denom
    return np.array([x, y])


def save_aggregate_eutectic_plot(eut_rows, output_path):
    if not eut_rows:
        return

    fig, ax = plt.subplots(figsize=(12, 11))

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
        [0.0, 0.0],
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linewidth=1.5)

    cmap = plt.get_cmap('tab20')
    for idx, row in enumerate(eut_rows):
        color = cmap(idx % 20)

        pred_abc = to_ternary_from_x0_x1(row['pred_x0'], row['pred_x1'])
        exp_abc = to_ternary_from_x0_x1(row['exp_x0'], row['exp_x1'])

        pred_xy = from_ternary_to_cartesian(pred_abc)
        exp_xy = from_ternary_to_cartesian(exp_abc)

        ax.scatter(exp_xy[0], exp_xy[1], marker='s', s=85, color=color, edgecolor='black', linewidth=0.5)
        ax.scatter(pred_xy[0], pred_xy[1], marker='o', s=85, color=color, edgecolor='black', linewidth=0.5)

        center = 0.5 * (pred_xy + exp_xy)
        vec = exp_xy - pred_xy
        dist = np.linalg.norm(vec)
        angle_deg = np.degrees(np.arctan2(vec[1], vec[0])) if dist > 0 else 0.0
        major = max(0.02, dist * 1.05)
        minor = max(0.004, major / 4.5)

        ellipse = Ellipse(
            xy=center,
            width=2.0 * major,
            height=2.0 * minor,
            angle=angle_deg,
            facecolor=color,
            edgecolor=color,
            alpha=0.16,
            linewidth=1.0,
        )
        ax.add_patch(ellipse)

        ax.plot([pred_xy[0], exp_xy[0]], [pred_xy[1], exp_xy[1]], color=color, linewidth=1.0, alpha=0.75)

        pred_label = f"{row['system']} P:{row['pred_temp_K']:.0f}K"
        exp_label = f"E:{row['exp_temp_K']:.0f}K"
        ax.text(pred_xy[0] + 0.008, pred_xy[1] + 0.008, pred_label, fontsize=7, color=color)
        ax.text(exp_xy[0] + 0.008, exp_xy[1] - 0.012, exp_label, fontsize=7, color=color)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.07)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title('Aggregate Eutectic Composition Agreement with Temperature Labels')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_misc_gap_temperature_plot(temp_rows, output_path):
    if not temp_rows:
        return

    rows_sorted = sorted(temp_rows, key=lambda d: d['system'])
    x = np.arange(len(rows_sorted))
    labels = [row['system'] for row in rows_sorted]
    tern_max = [row['max_ternary_K'] for row in rows_sorted]
    binary_max = [row['max_binary_K'] for row in rows_sorted]

    fig_width = max(12, len(rows_sorted) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.scatter(x - 0.12, binary_max, color='tab:blue', marker='s', s=55, label='Max Binary Reference Temp (K)')
    ax.scatter(x + 0.12, tern_max, color='tab:red', marker='o', s=55, label='Max Ternary Interior Liquidus Temp (K)')

    ax.set_ylabel('Temperature (K)')
    ax.set_title('Binary-Reference vs Max Ternary Liquidus Temperature by System')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"

    # ----------------------------
    # EDIT PATHS IN THIS SECTION
    # ----------------------------
    test_prefix = "B"
    work_dir = "all_dumps/ternary_test"
    binary_fit_param_path = "data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx"
    binary_pred_param_path = "data/ternary_dft_data/final_ml_params-internal.xlsx"
    test_set_filename = "test_set.xlsx"
    # ----------------------------

    os.makedirs(work_dir, exist_ok=True)


    '''
    ||||||||||||||||||||||||||||||||||||||||
    |||  SPECIFY TEST L PARAMETERS HERE  |||
    ||||||||||||||||||||||||||||||||||||||||
    '''

    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_df = pd.read_excel(binary_fit_param_path)

    ternary_df_path = os.path.join(work_dir, test_set_filename)
    assert os.path.exists(ternary_df_path), f"Ternary test file {ternary_df_path} does not exist in working directory."

    ternary_df = pd.read_excel(ternary_df_path)
    ternary_sys_list = ternary_df["test_system"].tolist()
    ternary_sys_list = [ast.literal_eval(e) if isinstance(e, str) else e for e in ternary_sys_list]

    results = {}
    inter_parity_rows = []
    eutectic_rows = []
    temp_check_rows = []

    absolute_temp_threshold_K = 4000.0
    deviation_threshold_K = 500.0

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
            missing_binary_params = False
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                else:
                    print(f"[WARN] Skipping {'_'.join(tern_sys)}: binary parameters not found for {bin_sys}.")
                    missing_binary_params = True
                    break

                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])

                if order_changed:
                    # Flip L1 parameter signs when element order is reversed
                    L1_a = -L1_a
                    L1_b = -L1_b

                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            if missing_binary_params:
                continue

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
            add_liquid_scatter_trace(tern_fig, plotter)

            ploff.plot(
                tern_fig,
                filename=os.path.join(work_dir, f"{test_prefix}_ternary_test_{'_'.join(tern_sys)}.html"),
                auto_open=False
            )

            melting_temps = plotter.get_inter_melting_temps([inter_phase])
            melting_temp = melting_temps[inter_phase] + 273.15
        
            delta_T = melting_temp - ternary_df.iloc[i]["melting_temp_K"]

            # Check liquidus surface continuity
            continuity_result = check_liquidus_continuity(
                plotter.liq_plotting_df, 
                delta=0.025,  # Same delta as used in plotter
                gap_threshold=0.1  # Significant gap threshold
            )

            temp_checks = compute_liquidus_temperature_checks(plotter.liq_plotting_df, plotter.plotting_df)
            max_ternary_interior = temp_checks['max_ternary_interior_liquidus_K']
            max_binary_reference = temp_checks['max_binary_reference_temp_K']
            ternary_binary_deviation = temp_checks['ternary_binary_max_deviation_K']

            absolute_high_temp_flag = (not np.isnan(max_ternary_interior)) and (max_ternary_interior > absolute_temp_threshold_K)
            binary_deviation_flag = (
                (not np.isnan(ternary_binary_deviation))
                and (abs(ternary_binary_deviation) > deviation_threshold_K)
            )
            
            # Format continuity data for results
            continuity_data = format_continuity_for_results(continuity_result)

            system_label = "_".join(tern_sys)
            inter_parity_rows.append({
                'system': system_label,
                'experimental_K': float(ternary_df.iloc[i]["melting_temp_K"]),
                'predicted_K': float(melting_temp),
            })
            temp_check_rows.append({
                'system': system_label,
                'max_ternary_K': max_ternary_interior,
                'max_binary_K': max_binary_reference,
            })
            
            results[system_label] = {
                "sys_type": sys_type,
                "predicted_melting_temp_K": melting_temp,
                "experimental_melting_temp_K": ternary_df.iloc[i]["melting_temp_K"],
                "delta_T_K": delta_T,
                "liquidus_continuity": continuity_data,
                "max_ternary_interior_liquidus_K": max_ternary_interior,
                "max_binary_reference_temp_K": max_binary_reference,
                "max_reference_phase_temp_K": temp_checks['max_reference_phase_temp_K'],
                "binary_edge_maxima_K": temp_checks['binary_edge_maxima_K'],
                "ternary_binary_max_deviation_K": ternary_binary_deviation,
                "absolute_high_temp_flag": absolute_high_temp_flag,
                "binary_deviation_flag": binary_deviation_flag,
                "temperature_check_notes": temp_checks['notes']
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
            missing_binary_params = False
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                else:
                    print(f"[WARN] Skipping {'_'.join(tern_sys)}: binary parameters not found for {bin_sys}.")
                    missing_binary_params = True
                    break

                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])

                if order_changed:
                    # Flip L1 parameter signs when element order is reversed
                    L1_a = -L1_a
                    L1_b = -L1_b

                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

            if missing_binary_params:
                continue

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
            add_liquid_scatter_trace(tern_fig, plotter)

            ploff.plot(
                tern_fig,
                filename=os.path.join(work_dir, f"{test_prefix}_ternary_test_{'_'.join(tern_sys)}.html"),
                auto_open=False
            )
            
            # Extract eutectic point
            eutectic_point = get_eutectic_point(plotter.liq_plotting_df)
            
            # Check liquidus surface continuity
            continuity_result = check_liquidus_continuity(
                plotter.liq_plotting_df, 
                delta=0.025,  # Same delta as used in plotter
                gap_threshold=0.1  # adjust to desired significant gap threshold
            )

            temp_checks = compute_liquidus_temperature_checks(plotter.liq_plotting_df, plotter.plotting_df)
            max_ternary_interior = temp_checks['max_ternary_interior_liquidus_K']
            max_binary_reference = temp_checks['max_binary_reference_temp_K']
            ternary_binary_deviation = temp_checks['ternary_binary_max_deviation_K']

            absolute_high_temp_flag = (not np.isnan(max_ternary_interior)) and (max_ternary_interior > absolute_temp_threshold_K)
            binary_deviation_flag = (
                (not np.isnan(ternary_binary_deviation))
                and (abs(ternary_binary_deviation) > deviation_threshold_K)
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

            system_label = "_".join(tern_sys)
            eutectic_rows.append({
                'system': system_label,
                'pred_x0': float(eutectic_point['x0']),
                'pred_x1': float(eutectic_point['x1']),
                'exp_x0': float(exp_x0),
                'exp_x1': float(exp_x1),
                'pred_temp_K': float(eutectic_point['temperature_K']),
                'exp_temp_K': float(ternary_df.iloc[i]["eutectic_temp_K"]),
            })
            temp_check_rows.append({
                'system': system_label,
                'max_ternary_K': max_ternary_interior,
                'max_binary_K': max_binary_reference,
            })
            
            results[system_label] = {
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
                "liquidus_continuity": continuity_data,
                "max_ternary_interior_liquidus_K": max_ternary_interior,
                "max_binary_reference_temp_K": max_binary_reference,
                "max_reference_phase_temp_K": temp_checks['max_reference_phase_temp_K'],
                "binary_edge_maxima_K": temp_checks['binary_edge_maxima_K'],
                "ternary_binary_max_deviation_K": ternary_binary_deviation,
                "absolute_high_temp_flag": absolute_high_temp_flag,
                "binary_deviation_flag": binary_deviation_flag,
                "temperature_check_notes": temp_checks['notes']
            }

    parity_path = os.path.join(work_dir, f"{test_prefix}_intermetallic_temp_parity.png")
    eutectic_path = os.path.join(work_dir, f"{test_prefix}_eutectic_aggregate.png")
    temp_check_path = os.path.join(work_dir, f"{test_prefix}_misc_gap_temperature_check.png")

    save_intermetallic_parity_plot(inter_parity_rows, parity_path)
    save_aggregate_eutectic_plot(eutectic_rows, eutectic_path)
    save_misc_gap_temperature_plot(temp_check_rows, temp_check_path)

    absolute_flagged = [sys for sys, data in results.items() if data.get('absolute_high_temp_flag', False)]
    deviation_flagged = [sys for sys, data in results.items() if data.get('binary_deviation_flag', False)]

    pprint.pprint(results, sort_dicts=False)
    print(f"\nSaved diagnostic figure: {parity_path}")
    print(f"Saved diagnostic figure: {eutectic_path}")
    print(f"Saved diagnostic figure: {temp_check_path}")
    print(f"Absolute high-temp flags (> {absolute_temp_threshold_K:.0f} K): {len(absolute_flagged)}")
    if absolute_flagged:
        print(f"  Systems: {absolute_flagged}")
    print(f"Ternary-vs-binary deviation flags (> {deviation_threshold_K:.0f} K): {len(deviation_flagged)}")
    if deviation_flagged:
        print(f"  Systems: {deviation_flagged}")

    # dump results to json file
    results_file = os.path.join(work_dir, f"{test_prefix}_ternary_test_report.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

                


if __name__ == "__main__":
    main()