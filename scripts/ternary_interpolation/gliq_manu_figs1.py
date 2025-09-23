import pandas as pd 
import matplotlib.pyplot as plt
import ast
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from adjustText import adjust_text
import json

def eutectic_fig():
    eut_path = "all_dumps/gliq_manu_test_eut/ternary_eutectic_results.xlsx"

    df_eut = pd.read_excel(eut_path)
    # columns_to_drop = ['Reference:']
    # df_eut = df_eut.drop(columns=columns_to_drop)
    # drop all rows where 'Eutectic Composition' is NaN
    print(df_eut)
    # exit()
    df_eut = df_eut.dropna(subset=['Experimental Eut (K)'])


    print(df_eut)
    # convert the lists in Ternary column to lists using ast.literal_eval
    df_eut['Ternary'] = df_eut['Ternary'].apply(lambda x: ast.literal_eval(x))
    df_eut['Eutectic Composition'] = df_eut['Eutectic Composition'].apply(lambda x: ast.literal_eval(x))
    df_eut['Exp Eut Composition'] = df_eut['Exp Eut Composition'].apply(lambda x: ast.literal_eval(x))

    plt.rcParams['pdf.fonttype'] = 42
    print(plt.rcParams['pdf.fonttype'])

    df = df_eut.copy()

    # Generate system labels like 'Cd-Sn-Tl'
    df['System'] = df['Ternary'].apply(lambda x: '-'.join(x))

    # Compute absolute temperature difference
    df['Abs Temp Diff'] = abs(df['Eutectic Temp (K)'] - df['Experimental Eut (K)'])

    # Plot 1: Overlayed bar plot of simulated and experimental eutectic temps
    x = np.arange(len(df))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, df['Eutectic Temp (K)'], width, label='Interpolated')
    ax1.bar(x + width/2, df['Experimental Eut (K)'], width, label='Published')
    ax1.set_ylabel('Eutectic Temperature (K)')
    # ax1.set_title('Simulated vs Experimental Eutectic Temperatures')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['System'], rotation=45)
    ax1.legend()
    plt.tight_layout()
    # plt.show()



    def to_ternary(xy):
        x, y = xy
        return [1 - x - y, x, y]

    def from_ternary(abc):
        # Projects ternary (a, b, c) -> 2D (x, y) using equilateral triangle projection
        a, b, c = abc
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = (np.sqrt(3) / 2) * c / (a + b + c)
        return np.array([x, y])

    def to_ternary_coords(xy):
        # Inverse projection (approximate, assuming sum to 1)
        x, y = xy
        c = y * 2 / np.sqrt(3)
        b = x - c / 2
        a = 1 - b - c
        return [a, b, c]



    df['System'] = df['Ternary'].apply(lambda x: '-'.join(x))
    df['Eut_A_B_C'] = df['Eutectic Composition'].apply(to_ternary)
    df['Exp_A_B_C'] = df['Exp Eut Composition'].apply(to_ternary)

    colors = px.colors.qualitative.Plotly
    systems = df['System'].unique()
    color_map = {sys: colors[i % len(colors)] for i, sys in enumerate(systems)}

    triangle_trace = go.Scatterternary(
        mode='lines',
        a=[1, 0, 0, 1],
        b=[0, 1, 0, 0],
        c=[0, 0, 1, 0],
        line=dict(color='black', width=2),
        showlegend=False
    )

    # Traces
    eutectic_traces, exp_traces, ellipse_traces = [], [], []

    marker_size = 40

    for _, row in df.iterrows():
        sys = row['System']
        color = color_map[sys]

        eut = row['Eut_A_B_C']
        exp = row['Exp_A_B_C']

        # Markers with increased size and some transparency for better visibility
        eutectic_traces.append(go.Scatterternary(
            a=[eut[0]], b=[eut[1]], c=[eut[2]],
            mode='markers', name=f'{sys} (Simulated)',
            marker=dict(color=color, symbol='circle', size=marker_size, opacity=0.8,
                    line=dict(width=2, color='white'))
        ))
        exp_traces.append(go.Scatterternary(
            a=[exp[0]], b=[exp[1]], c=[exp[2]],
            mode='markers', name=f'{sys} (Experimental)',
            marker=dict(color=color, symbol='square', size=marker_size, opacity=0.8,
                    line=dict(width=2, color='white'))
        ))

        # Ellipse around the two points
        p1 = from_ternary(eut)
        p2 = from_ternary(exp)
        center = (p1 + p2) / 2
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        angle = np.arctan2(vec[1], vec[0])

        # Ellipse parameters
        a = dist / 2 * 1.05  # semi-major axis
        b = a / 4.5           # semi-minor axis, controls "narrowness"
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_xy = np.stack([a * np.cos(t), b * np.sin(t)])

        # Rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        rotated = R @ ellipse_xy
        shifted = rotated.T + center

        # Back to ternary
        tern_points = np.array([to_ternary_coords(p) for p in shifted])
        tern_points = tern_points[(tern_points >= 0).all(axis=1)]  # valid triangle points

        ellipse_traces.append(go.Scatterternary(
            a=tern_points[:, 0], b=tern_points[:, 1], c=tern_points[:, 2],
            mode='lines', fill='toself',
            line=dict(color=color),
            fillcolor=color,
            opacity=0.1,
            showlegend=True,
        ))

    # Combine and show - order matters for layering (later traces appear on top)
    # Order: triangle -> ellipses -> squares (experimental) -> circles (simulated) to put circles in foreground
    fig = go.Figure([triangle_trace] + ellipse_traces + exp_traces + eutectic_traces)

    wd = ht = 1600
    scaler = 50
    sz = wd / scaler


    fig.update_layout(
        ternary=dict(
            sum=1,
            aaxis=dict(title='', min=0.0, linecolor='black', linewidth=4, 
                    tickfont=dict(size=sz), showgrid=True, gridcolor='gray', 
                    showticklabels=False),
            baxis=dict(title='', min=0.0, linecolor='black', linewidth=4,
                        tickfont=dict(size=sz), showgrid=True, gridcolor='gray',
                        showticklabels=False),
            caxis=dict(title='', min=0.0, linecolor='black', linewidth=4,
                        tickfont=dict(size=sz), showgrid=True, gridcolor='gray',
                        showticklabels=False),
            bgcolor='white',

        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        title='Ternary Diagram with Grouping Ellipses',
        legend=dict(itemsizing='constant', font=dict(size=10)),
        # Add margins to provide space around the ternary plot
        margin=dict(l=100, r=100, t=100, b=100)
    )


    fig.update_layout(
        width=wd,
        height=ht,
    )

    fig.show()


def inter_figure(): 
    inter_path = "all_dumps/gliq_manu_test3/ternary_Gliq_mps_final_linear.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)

    # evaluate elements column using ast.literal_eval
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create system identifier from elements list (A-B-C format)
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Check if system has "All fitted" fit type
    df['contains_pred'] = df['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('Fit Type') == 'Contains predicted'
    )

    print(df.head())
    # Color mapping
    colors = {'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'}
    df['color'] = df['type'].map(colors)

    # Create the scatter plot
    # plt.figure(figsize=(15, 4))
    plt.figure(figsize=(8, 6))
    
    # Plot points with different edge colors based on fit type
    for fit_type in [False, True]:  # Plot non-fitted first, then fitted (so fitted are on top)
        subset = df[df['contains_pred'] == fit_type]
        if len(subset) > 0:
            edge_color = 'black' if fit_type else 'none'
            edge_width = 2 if fit_type else 0
            plt.scatter(
                subset['melting_point_k'],
                subset['gliq_melting_temp'],
                c=subset['color'],
                s=80,  # Adjust marker size here (default is 20)
                edgecolors=edge_color,
                linewidths=edge_width
            )

    # Toggle label by commenting/uncommenting
    # texts = []
    # for _, row in df.iterrows():
    #     texts.append(plt.text(row['melting_point_k'] + 5, row['gliq_melting_temp'], row['reduced_formula'], fontsize=8))

    # # Adjust text to reduce overlaps
    # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Plot the reference y=x line
    min_val = min(df['melting_point_k'].min(), df['gliq_melting_temp'].min())
    max_val = max(df['melting_point_k'].max(), df['gliq_melting_temp'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')

    # Axis labels and title
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')

    # modify x-axis limits
    # plt.xlim(min_val, max_val - 150)
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)

    # # Custom legend for congruency
    # handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, label=label)
    #            for label, color in colors.items()]
    # handles.append(plt.Line2D([], [], linestyle='--', color='k', label='y = x'))
    # plt.legend(handles=handles)

    # plt.grid(True)
    plt.tight_layout()
    plt.show()


def inter_figure_filtered():
    inter_path = "all_dumps/gliq_manu_test2/ternary_Gliq_mps_final_linear.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)

    # evaluate elements column using ast.literal_eval
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create system identifier from elements list (A-B-C format)
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Check if system has "Contains predicted" fit type
    df['contains_pred'] = df['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('Fit Type') == 'Contains predicted'
    )

    # Filter out points that would have black edges (Contains predicted)
    df_filtered = df[df['contains_pred'] == False].copy()
    
    print(f"Original points: {len(df)}, Filtered points: {len(df_filtered)}")
    
    # Color mapping
    colors = {'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'}
    df_filtered['color'] = df_filtered['type'].map(colors)

    # Create the second scatter plot (filtered)
    plt.figure(figsize=(8, 6))
    
    plt.scatter(
        df_filtered['melting_point_k'],
        df_filtered['gliq_melting_temp'],
        c=df_filtered['color'],
        s=80,  # Adjust marker size here (default is 20)
        edgecolors='none',
        linewidths=0
    )

    # Plot the reference y=x line
    min_val = min(df_filtered['melting_point_k'].min(), df_filtered['gliq_melting_temp'].min())
    max_val = max(df_filtered['melting_point_k'].max(), df_filtered['gliq_melting_temp'].max()) 
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')

    # Axis labels and title
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')

    # modify x-axis limits
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)

    plt.tight_layout()
    plt.show()


def inter_figure_error_metrics():
    inter_path = "all_dumps/gliq_manu_test2/ternary_Gliq_mps_final_linear.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)

    # evaluate elements column using ast.literal_eval
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create system identifier from elements list (A-B-C format)
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Check if system has "Contains predicted" fit type
    df['contains_pred'] = df['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('Fit Type') == 'Contains predicted'
    )

    # Filter out points that would have black edges (Contains predicted)
    df_filtered = df[df['contains_pred'] == False].copy()
    
    # Extract MAE and RMSE values and compute averages
    df_filtered['mae_avg'] = df_filtered['system_key'].apply(
        lambda x: np.mean(meta_data.get(x, {}).get('norm_mae', [0]))
    )
    df_filtered['rmse_avg'] = df_filtered['system_key'].apply(
        lambda x: np.mean(meta_data.get(x, {}).get('norm_rmse', [0]))
    )
    
    # Color mapping
    colors = {'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'}
    df_filtered['color'] = df_filtered['type'].map(colors)
    
    # Scale point sizes based on error metrics (normalize to reasonable range)
    def scale_sizes(values, min_size=20, max_size=200):
        if values.max() == values.min():
            return np.full_like(values, min_size)
        normalized = (values - values.min()) / (values.max() - values.min())
        return min_size + normalized * (max_size - min_size)
    
    mae_sizes = scale_sizes(df_filtered['mae_avg'])
    rmse_sizes = scale_sizes(df_filtered['rmse_avg'])
    
    # Create MAE plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_filtered['melting_point_k'],
        df_filtered['gliq_melting_temp'],
        c=df_filtered['color'],
        s=mae_sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Plot the reference y=x line
    min_val = min(df_filtered['melting_point_k'].min(), df_filtered['gliq_melting_temp'].min())
    max_val = max(df_filtered['melting_point_k'].max(), df_filtered['gliq_melting_temp'].max()) - 300
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)
    plt.tight_layout()
    plt.show()
    
    # Create RMSE plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_filtered['melting_point_k'],
        df_filtered['gliq_melting_temp'],
        c=df_filtered['color'],
        s=rmse_sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Plot the reference y=x line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"MAE range: {df_filtered['mae_avg'].min():.2f} - {df_filtered['mae_avg'].max():.2f}")
    print(f"RMSE range: {df_filtered['rmse_avg'].min():.2f} - {df_filtered['rmse_avg'].max():.2f}")


def ternary_hull_metrics(include_predicted=True):
    """
    Analyze ternary hull metrics from metadata JSON.
    
    Parameters:
    include_predicted (bool): If True, include "Contains predicted" systems. 
                             If False, filter them out.
    """
    import json
    
    inter_path = "all_dumps/gliq_manu_test3/ternary_Gliq_mps_final_linear.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)

    # evaluate elements column using ast.literal_eval
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create system identifier from elements list (A-B-C format)
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Check if system has "Contains predicted" fit type
    df['contains_pred'] = df['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('Fit Type') == 'Contains predicted'
    )

    # Apply filtering based on toggle
    if include_predicted:
        df_filtered = df.copy()
        title_suffix = "All Systems"
    else:
        df_filtered = df[df['contains_pred'] == False].copy()
        title_suffix = "Filtered: Excluding 'Contains predicted'"
    
    print(f"Total systems: {len(df)}, Plotting: {len(df_filtered)}")
    
    # Extract ternary hull metrics
    df_filtered['n_ternary_compounds'] = df_filtered['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('ternary_meta', {}).get('n_ternary_compounds', 0)
    )
    df_filtered['deepest_formation_energy'] = df_filtered['system_key'].apply(
        lambda x: meta_data.get(x, {}).get('ternary_meta', {}).get('deepest_formation_energy', 0)
    )
    
    # Convert deepest formation energy to absolute values for better visualization
    df_filtered['abs_deepest_formation_energy'] = np.abs(df_filtered['deepest_formation_energy'])
    
    # Color mapping
    colors = {'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'}
    df_filtered['color'] = df_filtered['type'].map(colors)
    
    # Scale point sizes based on metrics (normalize to reasonable range)
    def scale_sizes(values, min_size=20, max_size=200):
        if values.max() == values.min():
            return np.full_like(values, min_size)
        normalized = (values - values.min()) / (values.max() - values.min())
        return min_size + normalized * (max_size - min_size)
    
    n_compounds_sizes = scale_sizes(df_filtered['n_ternary_compounds'])
    formation_energy_sizes = scale_sizes(df_filtered['abs_deepest_formation_energy'])
    
    # Create n_ternary_compounds plot
    plt.figure(figsize=(8, 6))
    scatter1 = plt.scatter(
        df_filtered['melting_point_k'],
        df_filtered['gliq_melting_temp'],
        c=df_filtered['color'],
        s=n_compounds_sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Plot the reference y=x line
    min_val = min(df_filtered['melting_point_k'].min(), df_filtered['gliq_melting_temp'].min())
    max_val = max(df_filtered['melting_point_k'].max(), df_filtered['gliq_melting_temp'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x', alpha=0.5)
    
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')
    plt.title(f'Point Size by Number of Ternary Compounds - {title_suffix}')
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)
    
    # Add colorbar legend for point sizes
    handles, labels = [], []
    for compound_count in sorted(df_filtered['n_ternary_compounds'].unique()):
        if compound_count > 0:  # Only show non-zero counts
            size = scale_sizes(np.array([compound_count]), min_size=20, max_size=200)[0]
            handles.append(plt.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', linewidths=0.5))
            labels.append(f'{int(compound_count)} compounds')
    
    plt.tight_layout()
    plt.show()
    
    # Create deepest formation energy plot
    plt.figure(figsize=(8, 6))
    scatter2 = plt.scatter(
        df_filtered['melting_point_k'],
        df_filtered['gliq_melting_temp'],
        c=df_filtered['color'],
        s=formation_energy_sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Plot the reference y=x line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x', alpha=0.5)
    
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')
    plt.title(f'Point Size by |Deepest Formation Energy| - {title_suffix}')
    plt.xlim(300, 2500)
    plt.ylim(300, 2500)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Number of ternary compounds range: {df_filtered['n_ternary_compounds'].min():.0f} - {df_filtered['n_ternary_compounds'].max():.0f}")
    print(f"Deepest formation energy range: {df_filtered['deepest_formation_energy'].min():.0f} - {df_filtered['deepest_formation_energy'].max():.0f}")
    print(f"|Deepest formation energy| range: {df_filtered['abs_deepest_formation_energy'].min():.0f} - {df_filtered['abs_deepest_formation_energy'].max():.0f}")


def main():
    # eutectic_fig()
    inter_figure()
    inter_figure_filtered()
    inter_figure_error_metrics()
    
    # Run hull metrics analysis
    print("\n=== Ternary Hull Metrics: All Systems ===")
    ternary_hull_metrics(include_predicted=True)
    
    # print("\n=== Ternary Hull Metrics: Filtered Systems ===")
    # ternary_hull_metrics(include_predicted=False)


if __name__ == "__main__":
    main()