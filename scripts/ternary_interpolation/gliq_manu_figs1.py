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
    # inter_path = "all_dumps/gliq_manu_test4/ternary_Gliq_mps_final_linear.xlsx"
    # meta_data_path = "all_dumps/gliq_manu_test4/ternary_Gliq_meta_final_linear.json"

    inter_path = "all_dumps/gliq_manu_test7_linear/ternary_Gliq_mps_final_linear_updated.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test7_linear/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)
    # df = pd.read_csv(inter_path)

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
    
    # Calculate and print RMSE between interpolated and MPDS values
    differences = df['gliq_melting_temp'] - df['melting_point_k']
    rmse = np.sqrt(np.mean(differences**2))
    mae = np.mean(np.abs(differences))
    std_dev = np.std(differences)
    print(f"Average RMSE between interpolated and MPDS values: {rmse:.2f} K")
    print(f"Average MAE between interpolated and MPDS values: {mae:.2f} K")
    print(f"Standard deviation of differences: {std_dev:.2f} K")
    
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
    texts = []
    for _, row in df.iterrows():
        texts.append(plt.text(row['melting_point_k'] + 5, row['gliq_melting_temp'], row['reduced_formula'], fontsize=8))

    # Adjust text to reduce overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

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
    inter_path = "all_dumps/gliq_manu_test7_linear/ternary_Gliq_mps_final_linear_updated.xlsx"
    meta_data_path = "all_dumps/gliq_manu_test7_linear/ternary_Gliq_meta_final_linear.json"

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

    print(df_filtered)
    print(df_filtered["reduced_formula"].to_list())

    print(len(df_filtered))
    
    print(f"Original points: {len(df)}, Filtered points: {len(df_filtered)}")
    
    # Calculate and print RMSE between interpolated and MPDS values for filtered data
    differences_filtered = df_filtered['gliq_melting_temp'] - df_filtered['melting_point_k']
    rmse_filtered = np.sqrt(np.mean(differences_filtered**2))
    mae_filtered = np.mean(np.abs(differences_filtered))
    std_dev_filtered = np.std(differences_filtered)
    
    # compute standard dev from formula and not using np.std
    n = len(differences_filtered)
    mean_diff = np.mean(differences_filtered)
    variance = np.sum((differences_filtered - mean_diff)**2) / n
    std_dev_filtered_formula = np.sqrt(variance)

    print(f"Average RMSE between interpolated and MPDS values (filtered): {rmse_filtered:.2f} K")
    print(f"Average MAE between interpolated and MPDS values (filtered): {mae_filtered:.2f} K")
    print(f"Standard deviation of differences (filtered): {std_dev_filtered:.2f} K")
    print(f"Standard deviation of differences (filtered, formula): {std_dev_filtered_formula:.2f} K")
    
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
    # Toggle label by commenting/uncommenting
    texts = []
    for _, row in df.iterrows():
        texts.append(plt.text(row['melting_point_k'] + 5, row['gliq_melting_temp'], row['reduced_formula'], fontsize=8))

    # Adjust text to reduce overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

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


def binary_L_parameter_analysis(include_predicted=True, deviation_metric='mad'):
    """
    Analyze variation in binary L parameters across the three binary systems in each ternary.
    
    Parameters:
    include_predicted (bool): If True, include "Contains predicted" systems. 
                             If False, filter them out.
    deviation_metric (str): Metric to measure parameter variation. Options:
                           'mad' - Mean Absolute Deviation
                           'std' - Standard Deviation
                           'range' - Range (max - min)
                           'pairwise' - Pairwise distance metric
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
    
    def calculate_parameter_deviation(system_key, param_index, metric):
        """Calculate deviation metric for a specific L parameter across three binaries"""
        binary_params = meta_data.get(system_key, {}).get('binary_L_params', {})
        if not binary_params:
            return 0.0
        
        # Extract the parameter values (first 3 values from each binary system)
        param_values = []
        for binary_key, params in binary_params.items():
            if len(params) >= 4:  # Ensure we have all 4 parameters
                param_values.append(params[param_index])
        
        if len(param_values) < 3:  # Need all 3 binary systems
            return 0.0
        
        param_array = np.array(param_values)
        
        # Calculate different deviation metrics
        if metric == 'mad':
            # Mean Absolute Deviation
            mean_val = np.mean(param_array)
            deviation = np.mean(np.abs(param_array - mean_val))
        elif metric == 'std':
            # Standard Deviation
            deviation = np.std(param_array, ddof=0)  # Population std
        elif metric == 'range':
            # Range (max - min)
            deviation = np.max(param_array) - np.min(param_array)
        elif metric == 'pairwise':
            # Pairwise distance metric: sqrt(sum of squared differences between pairs)
            x1, x2, x3 = param_array[0], param_array[1], param_array[2]
            deviation = np.sqrt((x1-x2)**2 + (x1-x3)**2 + (x2-x3)**2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return deviation
    
    # Calculate deviation metric for each L parameter
    metric_suffix = deviation_metric.upper()
    df_filtered[f'L0_a_{deviation_metric}'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_deviation(x, 0, deviation_metric))
    df_filtered[f'L0_b_{deviation_metric}'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_deviation(x, 1, deviation_metric))
    df_filtered[f'L1_b_{deviation_metric}'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_deviation(x, 2, deviation_metric))
    
    # Color mapping
    colors = {'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'}
    df_filtered['color'] = df_filtered['type'].map(colors)
    
    # Scale point sizes based on MAD values (normalize to reasonable range)
    def scale_sizes(values, min_size=20, max_size=200):
        if values.max() == values.min() or values.max() == 0:
            return np.full_like(values, min_size)
        normalized = (values - values.min()) / (values.max() - values.min())
        return min_size + normalized * (max_size - min_size)
    
    # Create plots for each L parameter
    metric_names = {
        'mad': 'Mean Absolute Deviation',
        'std': 'Standard Deviation', 
        'range': 'Range',
        'pairwise': 'Pairwise Distance'
    }
    metric_name = metric_names.get(deviation_metric, deviation_metric.upper())
    
    l_params = [
        (f'L0_a_{deviation_metric}', 'L0_a', f'L0_a ({metric_name})'),
        (f'L0_b_{deviation_metric}', 'L0_b', f'L0_b ({metric_name})'),
        (f'L1_b_{deviation_metric}', 'L1_b', f'L1_a ({metric_name})')
    ]
    
    for param_col, param_name, param_title in l_params:
        sizes = scale_sizes(df_filtered[param_col])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_filtered['melting_point_k'],
            df_filtered['gliq_melting_temp'],
            c=df_filtered['color'],
            s=sizes,
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
        plt.title(f'Point Size by {param_title} Variation - {title_suffix}')
        plt.xlim(300, 2500)
        plt.ylim(300, 2500)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics for this parameter
        print(f"{param_name} {metric_name} range: {df_filtered[param_col].min():.2f} - {df_filtered[param_col].max():.2f}")
    
    # Print overall statistics
    print(f"\nSummary for {title_suffix} using {metric_name}:")
    print(f"Systems with complete binary L parameters: {len(df_filtered[df_filtered[f'L0_a_{deviation_metric}'] > 0])}")
    
    # Add plots for average magnitude of each L parameter
    def calculate_parameter_avg_magnitude(system_key, param_index):
        """Calculate average magnitude for a specific L parameter across three binaries"""
        binary_params = meta_data.get(system_key, {}).get('binary_L_params', {})
        if not binary_params:
            return 0.0
        
        # Extract the parameter values (first 3 values from each binary system)
        param_values = []
        for binary_key, params in binary_params.items():
            if len(params) >= 4:  # Ensure we have all 4 parameters
                param_values.append(abs(params[param_index]))  # Take absolute value for magnitude
        
        if len(param_values) < 3:  # Need all 3 binary systems
            return 0.0
        
        # Calculate average magnitude
        avg_magnitude = np.mean(param_values)
        return avg_magnitude
    
    # Calculate average magnitude for each L parameter
    df_filtered['L0_a_avg_mag'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_avg_magnitude(x, 0))
    df_filtered['L0_b_avg_mag'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_avg_magnitude(x, 1))
    df_filtered['L1_b_avg_mag'] = df_filtered['system_key'].apply(lambda x: calculate_parameter_avg_magnitude(x, 2))
    
    # Create plots for average magnitude of each L parameter
    l_params_mag = [
        ('L0_a_avg_mag', 'L0_a', 'L0_a Average Magnitude'),
        ('L0_b_avg_mag', 'L0_b', 'L0_b Average Magnitude'),
        ('L1_b_avg_mag', 'L1_b', 'L1_b Average Magnitude')
    ]
    
    print(f"\n=== Average Magnitude Plots ===")
    
    for param_col, param_name, param_title in l_params_mag:
        sizes = scale_sizes(df_filtered[param_col])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_filtered['melting_point_k'],
            df_filtered['gliq_melting_temp'],
            c=df_filtered['color'],
            s=sizes,
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
        plt.title(f'Point Size by {param_title} - {title_suffix}')
        plt.xlim(300, 2500)
        plt.ylim(300, 2500)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics for this parameter
        print(f"{param_name} Average Magnitude range: {df_filtered[param_col].min():.2f} - {df_filtered[param_col].max():.2f}")
    
    print(f"\nAverage Magnitude Summary for {title_suffix}:")
    print(f"Systems with complete binary L parameters: {len(df_filtered[df_filtered['L0_a_avg_mag'] > 0])}")


def plot_L_parameter_distributions(meta_data_path="all_dumps/gliq_manu_test3/ternary_Gliq_meta_final_linear.json"):
    """
    Plot frequency distributions for L0_a, L0_b, and L1_b parameters across all entries in the metadata JSON.
    
    Parameters:
    meta_data_path (str): Path to the metadata JSON file
    """
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Collect all L parameter values across all systems and all binaries
    L0_a_values = []
    L0_b_values = []
    L1_b_values = []
    
    for system_key, system_data in meta_data.items():
        binary_params = system_data.get('binary_L_params', {})
        
        for binary_key, params in binary_params.items():
            if len(params) >= 4:  # Ensure we have all 4 parameters
                L0_a_values.append(params[0])  # L0_a
                L0_b_values.append(params[1])  # L0_b
                L1_b_values.append(params[2])  # L1_b (note: you mentioned L1_a but the data shows L1_b)
    
    # Convert to numpy arrays for easier handling
    L0_a_values = np.array(L0_a_values)
    L0_b_values = np.array(L0_b_values)
    L1_b_values = np.array(L1_b_values)
    
    print(f"Collected {len(L0_a_values)} L parameter sets from {len(meta_data)} ternary systems")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot L0_a distribution
    axes[0].hist(L0_a_values, bins=50, alpha=0.7, color='tab:blue', edgecolor='black')
    axes[0].set_xlabel('L0_a Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of L0_a Parameters')
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_L0_a = np.mean(L0_a_values)
    std_L0_a = np.std(L0_a_values)
    axes[0].axvline(mean_L0_a, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_L0_a:.1f}')
    axes[0].legend()
    
    # Plot L0_b distribution
    axes[1].hist(L0_b_values, bins=50, alpha=0.7, color='tab:orange', edgecolor='black')
    axes[1].set_xlabel('L0_b Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of L0_b Parameters')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_L0_b = np.mean(L0_b_values)
    std_L0_b = np.std(L0_b_values)
    axes[1].axvline(mean_L0_b, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_L0_b:.1f}')
    axes[1].legend()
    
    # Plot L1_b distribution
    axes[2].hist(L1_b_values, bins=50, alpha=0.7, color='tab:green', edgecolor='black')
    axes[2].set_xlabel('L1_b Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of L1_b Parameters')
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_L1_b = np.mean(L1_b_values)
    std_L1_b = np.std(L1_b_values)
    axes[2].axvline(mean_L1_b, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_L1_b:.1f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nL Parameter Statistics:")
    print(f"L0_a: Mean = {mean_L0_a:.2f}, Std = {std_L0_a:.2f}, Min = {np.min(L0_a_values):.2f}, Max = {np.max(L0_a_values):.2f}")
    print(f"L0_b: Mean = {mean_L0_b:.2f}, Std = {std_L0_b:.2f}, Min = {np.min(L0_b_values):.2f}, Max = {np.max(L0_b_values):.2f}")
    print(f"L1_b: Mean = {mean_L1_b:.2f}, Std = {std_L1_b:.2f}, Min = {np.min(L1_b_values):.2f}, Max = {np.max(L1_b_values):.2f}")
    
    # Create individual plots for better detail
    parameters = [
        (L0_a_values, 'L0_a', 'tab:blue'),
        (L0_b_values, 'L0_b', 'tab:orange'),
        (L1_b_values, 'L1_b', 'tab:green')
    ]
    
    for values, param_name, color in parameters:
        plt.figure(figsize=(8, 6))
        n, bins, patches = plt.hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
        
        plt.xlabel(f'{param_name} Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {param_name} Parameters')
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        # Add quartiles
        q25, q75 = np.percentile(values, [25, 75])
        plt.axvline(q25, color='orange', linestyle=':', linewidth=1, label=f'Q25: {q25:.1f}')
        plt.axvline(q75, color='orange', linestyle=':', linewidth=1, label=f'Q75: {q75:.1f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    # # eutectic_fig()
    inter_figure()
    inter_figure_filtered()
    # inter_figure_error_metrics()
    
    # # Run hull metrics analysis
    # ternary_hull_metrics(include_predicted=True)
    
    # ternary_hull_metrics(include_predicted=False)
    
    # Plot L parameter distributions
    # plot_L_parameter_distributions()
    
    # Run binary L parameter analysis with different deviation metrics
    # metrics = ['mad', 'std', 'range', 'pairwise']
    metrics = ['range']
    
    # for metric in metrics:
    #     print(f"\n{'='*50}")
    #     print(f"Binary L Parameter Analysis using {metric.upper()}")
    #     print(f"{'='*50}")
    #     print(f"\n=== All Systems ({metric.upper()}) ===")
    #     binary_L_parameter_analysis(include_predicted=True, deviation_metric=metric)
        

if __name__ == "__main__":
    main()