import pandas as pd 
import matplotlib.pyplot as plt
import ast
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from adjustText import adjust_text
import json
import re

def eutectic_fig():
    # eut_path = "all_dumps/gliq_manu_test_eut/ternary_eutectic_results_og.xlsx"
    eut_path = "all_dumps/gliq_manu_eut_new/ternary_eutectic_results.xlsx"

    df_eut = pd.read_excel(eut_path)
    # columns_to_drop = ['Reference:']
    # df_eut = df_eut.drop(columns=columns_to_drop)
    # drop all rows where 'Eutectic Composition' is NaN
    print(df_eut)
    # exit()
    # df_eut = df_eut.dropna(subset=['Experimental Eut (K)'])

    # drop all rows that contain NaN values in any of the columns
    df_eut = df_eut.dropna()
    print(df_eut)
    print(len(df_eut))


    print(df_eut)
    # convert the lists in Ternary column to lists using ast.literal_eval
    df_eut['Ternary'] = df_eut['Ternary'].apply(lambda x: ast.literal_eval(x))
    df_eut['Eutectic Composition'] = df_eut['Eutectic Composition'].apply(lambda x: ast.literal_eval(x))
    df_eut['Exp Eut Composition'] = df_eut['Exp Eut Composition'].apply(lambda x: ast.literal_eval(x))

    plt.rcParams['pdf.fonttype'] = 42
    print(plt.rcParams['pdf.fonttype'])

    df = df_eut.copy()

    # Ellipse tuning knobs:
    # - Increase ELLIPSE_BOUNDARY_TOL if near-edge systems lose their ellipse.
    # - Increase MIN_ELLIPSE_MAJOR_AXIS if small separations are hard to see.
    ELLIPSE_BOUNDARY_TOL = 1e-2
    MIN_ELLIPSE_MAJOR_AXIS = 0.012
    MIN_VALID_ELLIPSE_POINTS = 3
    DRAW_ELLIPSES_ON_TOP = True
    CIRCLE_MARKER_BORDER_WIDTH = 4.5

    # Per-system overrides for hard cases near corners/edges.
    # Keys can be either the original row label (e.g., "Al-Ga-Zn")
    # or a sorted label; both are checked.
    SYSTEM_ELLIPSE_OVERRIDES = {
        "Al-Ga-Zn": {
            "boundary_tol": 5e-2,
            "min_major_axis": 0.03,
            "min_valid_points": 1,
            "force_draw": True,
            "opacity": 0.25,
            "line_width": 2.5,
        }
    }

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

    # Error metrics after NaN row filtering.
    temp_differences = df['Eutectic Temp (K)'] - df['Experimental Eut (K)']
    eut_rmse = np.sqrt(np.mean(temp_differences ** 2))
    eut_mae = np.mean(np.abs(temp_differences))
    temp_std = np.std(temp_differences)

    # Composition distance uses straight-line distance in 2D Cartesian coordinates.
    # The dataset stores eutectic compositions as [x, y] points in triangle projection.
    eut_xy = np.vstack(df['Eutectic Composition'].to_numpy()).astype(float)
    exp_xy = np.vstack(df['Exp Eut Composition'].to_numpy()).astype(float)
    composition_distances = np.linalg.norm(eut_xy - exp_xy, axis=1)
    mean_composition_distance = np.mean(composition_distances*1)
    composition_std = np.std(composition_distances*1)

    print(f"Eutectic temperature RMSE (post-NaN filter): {eut_rmse:.2f} K")
    print(f"Eutectic temperature MAE (post-NaN filter): {eut_mae:.2f} K")
    print(f"Eutectic temperature std. dev. of differences (post-NaN filter): {temp_std:.2f} K")
    print(f"Average Cartesian composition distance (post-NaN filter): {mean_composition_distance:.4f}")
    print(f"Std. dev. of Cartesian composition distance (post-NaN filter): {composition_std:.4f}")



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
        sys_sorted = '-'.join(sorted(row['Ternary']))
        color = color_map[sys]

        boundary_tol = ELLIPSE_BOUNDARY_TOL
        min_major_axis = MIN_ELLIPSE_MAJOR_AXIS
        min_valid_points = MIN_VALID_ELLIPSE_POINTS
        force_draw = False
        ellipse_opacity = 0.16
        ellipse_line_width = 1.0

        override = SYSTEM_ELLIPSE_OVERRIDES.get(sys)
        if override is None:
            override = SYSTEM_ELLIPSE_OVERRIDES.get(sys_sorted)
        if override is not None:
            boundary_tol = float(override.get("boundary_tol", boundary_tol))
            min_major_axis = float(override.get("min_major_axis", min_major_axis))
            min_valid_points = int(override.get("min_valid_points", min_valid_points))
            force_draw = bool(override.get("force_draw", force_draw))
            ellipse_opacity = float(override.get("opacity", ellipse_opacity))
            ellipse_line_width = float(override.get("line_width", ellipse_line_width))

        eut = row['Eut_A_B_C']
        exp = row['Exp_A_B_C']

        # Markers with increased size and some transparency for better visibility
        eutectic_traces.append(go.Scatterternary(
            a=[eut[0]], b=[eut[1]], c=[eut[2]],
            mode='markers', name=f'{sys} (Simulated)',
            marker=dict(color=color, symbol='circle', size=marker_size, opacity=0.8,
                    line=dict(width=CIRCLE_MARKER_BORDER_WIDTH, color='black'))
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
        a = max(dist / 2 * 1.05, min_major_axis)  # semi-major axis
        b = a / 4.5           # semi-minor axis, controls "narrowness"
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_xy = np.stack([a * np.cos(t), b * np.sin(t)])

        # Rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        rotated = R @ ellipse_xy
        shifted = rotated.T + center

        # Back to ternary
        tern_points = np.array([to_ternary_coords(p) for p in shifted], dtype=float)

        # Soft-clip near-edge points instead of dropping them aggressively.
        # This helps preserve ellipses for systems near the triangle boundaries.
        tern_points = np.clip(tern_points, -boundary_tol, 1.0 + boundary_tol)
        tern_points = np.clip(tern_points, 0.0, 1.0)

        sums = tern_points.sum(axis=1)
        valid_sum_mask = sums > 1e-12
        tern_points = tern_points[valid_sum_mask]
        sums = sums[valid_sum_mask]
        if len(tern_points) < min_valid_points:
            if not force_draw:
                continue

            # Forced fallback: build a small visible polygon around the segment.
            if dist > 1e-12:
                u = vec / dist
            else:
                u = np.array([1.0, 0.0])
            v = np.array([-u[1], u[0]])

            p_a = center + u * min_major_axis + v * (min_major_axis / 6.0)
            p_b = center - u * min_major_axis + v * (min_major_axis / 6.0)
            p_c = center - v * (min_major_axis / 3.0)
            tern_points = np.array([to_ternary_coords(p_a), to_ternary_coords(p_b), to_ternary_coords(p_c)], dtype=float)
            tern_points = np.clip(tern_points, 0.0, 1.0)
            fallback_sums = tern_points.sum(axis=1, keepdims=True)
            fallback_sums[fallback_sums == 0.0] = 1.0
            tern_points = tern_points / fallback_sums
        tern_points = tern_points / sums[:, None]

        ellipse_traces.append(go.Scatterternary(
            a=tern_points[:, 0], b=tern_points[:, 1], c=tern_points[:, 2],
            mode='lines', fill='toself',
            line=dict(color=color, width=ellipse_line_width),
            fillcolor=color,
            opacity=ellipse_opacity,
            showlegend=True,
        ))

    # Combine and show - order matters for layering (later traces appear on top)
    if DRAW_ELLIPSES_ON_TOP:
        # Put ellipses above markers so tiny near-corner deviations remain visible.
        fig = go.Figure([triangle_trace] + exp_traces + eutectic_traces + ellipse_traces)
    else:
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

def inter_figure_raw():
    # inter_path = "all_dumps/gliq_manu/ternary_Gliq_mps.xlsx"
    inter_path = "all_dumps/gliq_manu_test4/ternary_Gliq_mps_final_linear.xlsx"
    df = pd.read_excel(inter_path)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['melting_point_k'], df['gliq_melting_temp'], c='blue', s=20, alpha=0.7)
    plt.xlabel('MPDS Congruent Melting Temperature (K)')
    plt.ylabel('Interpolated Melting Temperature (K)')

    plt.tick_params(axis='both', which='major', labelsize=12)

    min_val = min(df['melting_point_k'].min(), df['gliq_melting_temp'].min())
    max_val = max(df['melting_point_k'].max(), df['gliq_melting_temp'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')

    plt.xlim(500, 2500)
    plt.ylim(500, 2500)
    plt.tight_layout()
    plt.show()


def inter_figure(): 
    # Font settings - change these to customize appearance
    TICK_LABEL_SIZE = 15
    AXIS_LABEL_SIZE = 15
    LABEL_FONT_SIZE = 10
    AXIS_BORDER_WIDTH = 1.0
    LEGEND_FONT_SIZE = 14
    COLOR_BY_FIT_TYPE = True
    COLOR_BY_CONGRUENCY = False
    SHOW_LINEAR_FIT = False
    SINGLE_MARKER_COLOR = 'tab:blue'
    SHOW_COMPOUND_LABELS = True
    LABEL_MAX_COUNT = 18
    LABEL_NEAR_FIT_MIN_COUNT = 5
    LABEL_MIN_SPACING = 0.03
    LABEL_CONNECTOR_NEIGHBOR_RADIUS = 0.08
    LABEL_CONNECTOR_COLOR = '0.35'
    plt.rcParams['font.family'] = 'Arial'  # Set font to Arial
    
    # inter_path = "all_dumps/gliq_manu_test4/ternary_Gliq_mps_final_linear.xlsx"
    # meta_data_path = "all_dumps/gliq_manu_test4/ternary_Gliq_meta_final_linear.json"

    # inter_path = "all_dumps/gliq_manu_test_ultimate2/ternary_Gliq_mps_final_linear_updated.xlsx"
    # meta_data_path = "all_dumps/gliq_manu_test_ultimate2/ternary_Gliq_meta_final_linear.json"

    # inter_path = "all_dumps/gliq_manu_forreal/ternary_Gliq_mps_final_linear.xlsx"
    # meta_data_path = "all_dumps/gliq_manu_forreal/ternary_Gliq_meta_final_linear.json"

    inter_path = "all_dumps/gliq_manu_forreal_plusML/ternary_Gliq_mps_final_linear.xlsx"
    meta_data_path = "all_dumps/gliq_manu_forreal_plusML/ternary_Gliq_meta_final_linear.json"

    # inter_path = "all_dumps/gliq_manu_test4/ternary_Gliq_mps_final_linear.xlsx"
    # meta_data_path = "all_dumps/gliq_manu_test4/ternary_Gliq_meta_final_linear.json"

    df = pd.read_excel(inter_path)
    # df = pd.read_csv(inter_path)

    # evaluate elements column using ast.literal_eval
    df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(x))
    
    # Load metadata JSON
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create system identifier from elements list (A-B-C format)
    df['system_key'] = df['elements'].apply(lambda x: '-'.join(sorted(x)))
    
    # Check if system has "Contains predicted" fit type
    df['contains_pred'] = df.apply(
        lambda row: meta_data.get(
            f"{row['system_key']}__idx_{row['source_row_idx']}__phase_{row['reduced_formula']}",
            meta_data.get(row['system_key'], {})
        ).get('Fit Type') == 'Contains predicted',
        axis=1
    )

    print(df.head())

    # count the number of rows that are congruent and non-congruent under column name "type"
    print(df['type'].value_counts())

    print(f"Total ternaries: {len(df)}")
    print(f"All fitted ternaries: {(~df['contains_pred']).sum()}")
    print(f"Ternaries containing predicted binaries: {df['contains_pred'].sum()}")
    
    # Calculate and print RMSE between interpolated and MPDS values
    differences = df['gliq_melting_temp'] - df['melting_point_k']
    rmse = np.sqrt(np.mean(differences**2))
    mae = np.mean(np.abs(differences))
    mape = np.mean(np.abs(differences / df['melting_point_k'])) * 100
    std_dev = np.std(differences)
    print(f"Average RMSE between interpolated and MPDS values: {rmse:.2f} K")
    print(f"Average MAE between interpolated and MPDS values: {mae:.2f} K")
    print(f"Average MAPE between interpolated and MPDS values: {mape:.2f}%")
    print(f"Standard deviation of differences: {std_dev:.2f} K")

    for label, subset in [('All fitted', df[~df['contains_pred']]),
                          ('Contains predicted', df[df['contains_pred']])]:
        subset_differences = subset['gliq_melting_temp'] - subset['melting_point_k']
        subset_mae = np.mean(np.abs(subset_differences))
        subset_mape = np.mean(np.abs(subset_differences / subset['melting_point_k'])) * 100
        print(f"{label} MAE: {subset_mae:.2f} K; MAPE: {subset_mape:.2f}%")

    def add_compound_labels(ax, plot_df):
        def format_formula(formula):
            return re.sub(r'(\d+)', r'$_{\1}$', str(formula))

        label_df = plot_df.dropna(subset=['melting_point_k', 'gliq_melting_temp', 'reduced_formula']).copy()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        label_df = label_df[
            label_df['melting_point_k'].between(*sorted(xlim))
            & label_df['gliq_melting_temp'].between(*sorted(ylim))
        ].copy()
        if len(label_df) == 0:
            return

        x_span = max(abs(xlim[1] - xlim[0]), 1.0)
        y_span = max(abs(ylim[1] - ylim[0]), 1.0)
        label_df['abs_error'] = (label_df['gliq_melting_temp'] - label_df['melting_point_k']).abs()
        label_df['_nx'] = (label_df['melting_point_k'] - xlim[0]) / x_span
        label_df['_ny'] = (label_df['gliq_melting_temp'] - ylim[0]) / y_span
        label_df['_near_fit'] = label_df['abs_error'] / np.sqrt(x_span ** 2 + y_span ** 2)
        norm_points_by_index = {
            idx: np.array([row['_nx'], row['_ny']])
            for idx, row in label_df.iterrows()
        }

        near_fit_candidate_count = min(max(LABEL_NEAR_FIT_MIN_COUNT * 3, LABEL_MAX_COUNT // 3), len(label_df))
        near_fit_df = label_df.sort_values('_near_fit', ascending=True).head(near_fit_candidate_count)
        high_error_df = label_df.sort_values('abs_error', ascending=False).head(LABEL_MAX_COUNT)
        first_near_fit_count = min(LABEL_NEAR_FIT_MIN_COUNT * 2, len(near_fit_df))
        selected_df = pd.concat([
            near_fit_df.head(first_near_fit_count),
            high_error_df,
            near_fit_df.iloc[first_near_fit_count:],
        ])
        selected_df = selected_df[~selected_df.index.duplicated()]
        near_fit_candidate_indices = set(near_fit_df.index)
        selected = [row for _, row in selected_df.iterrows()]

        if len(selected) == 0:
            return

        def box_tuple(bbox):
            return bbox.x0, bbox.y0, bbox.x1, bbox.y1

        def boxes_overlap(a, b, pad=2):
            return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])

        def box_contains(box, point):
            return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

        def segment_intersects(a, b, c, d):
            def orientation(p, q, r):
                return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

            def on_segment(p, q, r):
                return (
                    min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                    and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
                )

            o1 = orientation(a, b, c)
            o2 = orientation(a, b, d)
            o3 = orientation(c, d, a)
            o4 = orientation(c, d, b)
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
            return (
                abs(o1) < 1e-9 and on_segment(a, c, b)
                or abs(o2) < 1e-9 and on_segment(a, d, b)
                or abs(o3) < 1e-9 and on_segment(c, a, d)
                or abs(o4) < 1e-9 and on_segment(c, b, d)
            )

        def segments_cross(a, b, c, d, margin=1e-3):
            r = np.array(b) - np.array(a)
            s = np.array(d) - np.array(c)
            denom = r[0] * s[1] - r[1] * s[0]
            if abs(denom) < 1e-9:
                return False
            diff = np.array(c) - np.array(a)
            t = (diff[0] * s[1] - diff[1] * s[0]) / denom
            u = (diff[0] * r[1] - diff[1] * r[0]) / denom
            return margin < t < 1 - margin and margin < u < 1 - margin

        def segment_intersects_box(segment, box):
            a, b = segment
            if box_contains(box, a) or box_contains(box, b):
                return True
            corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
            edges = list(zip(corners, corners[1:] + corners[:1]))
            return any(segment_intersects(a, b, c, d) for c, d in edges)

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = box_tuple(ax.get_window_extent(renderer))
        marker_points = ax.transData.transform(
            label_df[['melting_point_k', 'gliq_melting_temp']].astype(float).to_numpy()
        )
        marker_boxes = [(x - 10, y - 10, x + 10, y + 10) for x, y in marker_points]
        marker_box_by_index = dict(zip(label_df.index, marker_boxes))
        ref_min = max(min(xlim), min(ylim))
        ref_max = min(max(xlim), max(ylim))
        reference_segments = []
        if ref_min < ref_max:
            ref_disp = ax.transData.transform([(ref_min, ref_min), (ref_max, ref_max)])
            reference_segments.append((tuple(ref_disp[0]), tuple(ref_disp[1])))
        accepted_boxes = []
        accepted_segments = []
        direct_label_points = []
        accepted_count = 0
        accepted_near_fit_count = 0
        direct_offsets = [
            (12, 0, False), (-12, 0, False), (0, 12, False), (0, -12, False),
            (8, 0, False), (-8, 0, False), (0, 8, False), (0, -8, False),
        ]
        connector_offsets = [
            (-72, 48, True), (72, -48, True), (-72, -48, True), (72, 48, True),
            (-88, 0, True), (88, 0, True), (0, 62, True), (0, -62, True),
            (34, 20, True), (34, -20, True), (-34, 20, True), (-34, -20, True),
            (48, 0, True), (-48, 0, True), (18, 36, True), (18, -36, True),
            (-18, 36, True), (-18, -36, True),
        ]

        for row in selected:
            if accepted_count >= LABEL_MAX_COUNT and accepted_near_fit_count >= LABEL_NEAR_FIT_MIN_COUNT:
                break
            if accepted_count >= LABEL_MAX_COUNT and row.name not in near_fit_candidate_indices:
                continue

            point_data = (row['melting_point_k'], row['gliq_melting_temp'])
            point_disp = ax.transData.transform(point_data)
            own_marker_box = marker_box_by_index[row.name]
            other_marker_boxes = [box for idx, box in marker_box_by_index.items() if idx != row.name]
            point = norm_points_by_index[row.name]
            other_norm_points = [point for idx, point in norm_points_by_index.items() if idx != row.name]
            nearest_neighbor = min(
                (np.linalg.norm(point - other) for other in other_norm_points),
                default=np.inf
            )
            allow_direct_label = all(np.linalg.norm(point - other) >= LABEL_MIN_SPACING for other in direct_label_points)
            prefer_connector = row['_near_fit'] < LABEL_CONNECTOR_NEIGHBOR_RADIUS and nearest_neighbor < LABEL_CONNECTOR_NEIGHBOR_RADIUS
            point_offsets = (
                connector_offsets + (direct_offsets if allow_direct_label else [])
                if prefer_connector
                else (direct_offsets if allow_direct_label else []) + connector_offsets
            )
            point_offsets = [(*offset, False) for offset in point_offsets] + [(*offset, True) for offset in connector_offsets]

            for dx_pt, dy_pt, use_connector, allow_reference_crossing in point_offsets:
                offset_disp = np.array([dx_pt, dy_pt]) * fig.dpi / 72.0
                text_disp = point_disp + offset_disp
                text_data = ax.transData.inverted().transform(text_disp)
                text = ax.text(
                    text_data[0],
                    text_data[1],
                    format_formula(row['reduced_formula']),
                    fontsize=LABEL_FONT_SIZE,
                    fontweight='bold',
                    ha='left' if dx_pt > 0 else 'right' if dx_pt < 0 else 'center',
                    va='bottom' if dy_pt > 0 else 'top' if dy_pt < 0 else 'center',
                    zorder=5
                )
                fig.canvas.draw()
                text_box = box_tuple(text.get_window_extent(renderer).expanded(1.08, 1.15))
                segment = (tuple(point_disp), tuple(text_disp)) if use_connector else None
                clean = not any(boxes_overlap(text_box, box) for box in accepted_boxes + other_marker_boxes)
                clean = clean and not boxes_overlap(text_box, own_marker_box, pad=-2)
                clean = clean and not any(not (axes_box[0] <= text_box[i] <= axes_box[2]) for i in [0, 2])
                clean = clean and not any(not (axes_box[1] <= text_box[i] <= axes_box[3]) for i in [1, 3])
                clean = clean and not any(segment_intersects_box(other, text_box) for other in accepted_segments + reference_segments)

                if segment is not None:
                    blocked_segments = accepted_segments if allow_reference_crossing else accepted_segments + reference_segments
                    clean = clean and not any(segments_cross(segment[0], segment[1], other[0], other[1]) for other in blocked_segments)
                    clean = clean and not any(segment_intersects_box(segment, box) for box in accepted_boxes + other_marker_boxes)

                if clean:
                    accepted_boxes.append(text_box)
                    accepted_count += 1
                    if row.name in near_fit_candidate_indices:
                        accepted_near_fit_count += 1
                    if not use_connector:
                        direct_label_points.append(point)
                    if segment is not None:
                        accepted_segments.append(segment)
                        ax.annotate(
                            '',
                            xy=point_data,
                            xytext=text_data,
                            arrowprops=dict(arrowstyle='-', color=LABEL_CONNECTOR_COLOR, lw=0.8),
                            zorder=4
                        )
                    break

                text.remove()
    
    # Color mapping
    colors = {False: 'tab:blue', True: 'tab:orange'}
    if COLOR_BY_FIT_TYPE:
        df['color'] = df['contains_pred'].map(colors)
    elif COLOR_BY_CONGRUENCY:
        df['color'] = df['type'].map({'congruent': 'tab:cyan', 'non-congruent': 'tab:orange'})
    else:
        df['color'] = SINGLE_MARKER_COLOR

    # Create the scatter plot
    # plt.figure(figsize=(15, 4))
    plt.figure(figsize=(8, 6))
    
    # Plot all-fitted systems first, then systems containing predicted parameters
    for fit_type in [False, True]:
        subset = df[df['contains_pred'] == fit_type]
        if len(subset) > 0:
            plt.scatter(
                subset['melting_point_k'],
                subset['gliq_melting_temp'],
                c=subset['color'],
                s=80,  # Adjust marker size here (default is 20)
                edgecolors='none',
                linewidths=0,
                label=('Contains predicted binaries' if fit_type else 'All fitted binaries') if COLOR_BY_FIT_TYPE else None
            )

    # Plot the reference y=x line
    # min_val = min(df['melting_point_k'].min(), df['gliq_melting_temp'].min())
    # max_val = max(df['melting_point_k'].max(), df['gliq_melting_temp'].max())
    min_val = 775
    max_val = 2575
    # plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    plt.plot([min_val, max_val], [min_val, max_val], 'k--',)

    if SHOW_LINEAR_FIT:
        slope, intercept = np.polyfit(df['melting_point_k'], df['gliq_melting_temp'], 1)
        fit_x = np.array([df['melting_point_k'].min(), df['melting_point_k'].max()])
        plt.plot(fit_x, slope * fit_x + intercept, 'r--',
                 label=f'Linear fit: y = {slope:.2f}x {intercept:+.0f}')

    
    # Axis labels (bold, using variables defined at top of function)
    plt.xlabel('MPDS Congruent Melting Temperature (K)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    plt.ylabel('Predicted Melting Temperature (K)', fontsize=AXIS_LABEL_SIZE, fontweight='bold')

    # Tick labels (bold, using variable defined at top of function)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_BORDER_WIDTH)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # modify x-axis limits
    # plt.xlim(min_val, max_val - 150)
    plt.xlim(750, 2600)
    plt.ylim(750, 2600)

    # plt.xlim(500, 2500)
    # plt.ylim(500, 2500)

    if SHOW_COMPOUND_LABELS:
        add_compound_labels(ax, df)

    if COLOR_BY_FIT_TYPE or SHOW_LINEAR_FIT:
        plt.legend(fontsize=LEGEND_FONT_SIZE)

    # # Custom legend for marker colors
    # handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, label=label)
    #            for label, color in (colors.items() if COLOR_BY_CONGRUENCY else [('all', SINGLE_MARKER_COLOR)])]
    # handles.append(plt.Line2D([], [], linestyle='--', color='k', label='y = x'))
    # plt.legend(handles=handles)

    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/inter_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/inter_figure.svg', dpi=300, bbox_inches='tight')
    plt.show()


def inter_figure_filtered():
    inter_path = "all_dumps/gliq_manu_test7_linear/ternary_Gliq_mps_final_linear.xlsx"
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
        lambda x: meta_data.get(x, {}).get('Fit Type') == 'Contains predicted binaries'
    )

    # Filter out points that would have black edges (Contains predicted)
    df_filtered = df[df['contains_pred'] == False].copy()

    print(df_filtered)
    print(df_filtered["reduced_formula"].to_list())

    print(len(df_filtered))
    
    # dump filtered dataframe to xlsx
    df_filtered.to_excel("all_dumps/gliq_manu_test7_linear/ternary_Gliq_mps_final_linear_filtered.xlsx", index=False)

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
    # texts = []
    # for _, row in df.iterrows():
    #     texts.append(plt.text(row['melting_point_k'] + 5, row['gliq_melting_temp'], row['reduced_formula'], fontsize=8))

    # Adjust text to reduce overlaps
    # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

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

def inter_figure_correction():
    inter_path = "all_dumps/gliq_manu_forreal_plusML_correction/optimized_l0_tern_results.xlsx"
    
    # Load the optimization results
    df = pd.read_excel(inter_path)
    
    # Create the figure
    plt.figure(figsize=(9, 6))
    
    # Plot initial predicted temperatures (single color)
    plt.scatter(
        df['melting_point_k'],
        df['initial_gliq_temp'],
        c='gray',
        s=80,
        alpha=1.0,
        label='Initial prediction',
        zorder=2
    )
    
    # Draw connecting lines between initial and final predictions
    for _, row in df.iterrows():
        plt.plot(
            [row['melting_point_k'], row['melting_point_k']],
            [row['initial_gliq_temp'], row['final_gliq_temp']],
            'gray',
            linewidth=0.8,
            alpha=1.0,
            zorder=1
        )
    
    # Plot final corrected temperatures colored by l0_tern value
    scatter = plt.scatter(
        df['melting_point_k'],
        df['final_gliq_temp'],
        c=df['l0_tern'],
        s=80,
        cmap='RdBu_r',  # Diverging colormap: red (negative) -> white (0) -> blue (positive)
        edgecolors='black',
        linewidths=1,
        label='Corrected prediction',
        zorder=3,
        vmin=-abs(df['l0_tern']).max(),  # Center colormap at 0
        vmax=abs(df['l0_tern']).max()
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ternary L0-term (J/mol)', rotation=90, labelpad=20, fontweight='bold')
    
    # Plot the y=x reference line with slight extension beyond data range
    min_val = min(df['melting_point_k'].min(), 
                  df['initial_gliq_temp'].min(), 
                  df['final_gliq_temp'].min())
    max_val = max(df['melting_point_k'].max(), 
                  df['initial_gliq_temp'].max(), 
                  df['final_gliq_temp'].max())
    
    # Add a small margin (e.g., 2% of the range) to extend the line slightly
    margin = (max_val - min_val) * 0.02
    plt.plot([min_val - margin, max_val - 300], 
             [min_val - margin, max_val - 300], 
             'k--', zorder=0)
    
    # Axis labels and formatting
    plt.xlabel('MPDS Congruent Melting Temperature (K)', fontweight='bold')
    plt.ylabel('Interpolated Melting Temperature (K)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create second figure with only corrected results
    plt.figure(figsize=(9, 6))
    
    # Plot only final corrected temperatures colored by l0_tern value
    scatter = plt.scatter(
        df['melting_point_k'],
        df['final_gliq_temp'],
        c=df['l0_tern'],
        s=80,
        cmap='RdBu_r',  # Diverging colormap: red (negative) -> white (0) -> blue (positive)
        edgecolors='black',
        linewidths=1,
        label='Corrected prediction',
        zorder=3,
        vmin=-abs(df['l0_tern']).max(),  # Center colormap at 0
        vmax=abs(df['l0_tern']).max()
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ternary L0-term (J/mol)', rotation=90, labelpad=20, fontweight='bold')
    
    # Plot the y=x reference line with slight extension beyond data range
    min_val = min(df['melting_point_k'].min(), 
                  df['final_gliq_temp'].min())
    max_val = max(df['melting_point_k'].max(), 
                  df['final_gliq_temp'].max())
    
    # Add a small margin (e.g., 2% of the range) to extend the line slightly
    margin = (max_val - min_val) * 0.02
    plt.plot([min_val - margin, max_val - 300], 
             [min_val - margin, max_val - 300], 
             'k--', zorder=0)
    
    # Axis labels and formatting
    plt.xlabel('MPDS Congruent Melting Temperature (K)', fontweight='bold')
    plt.ylabel('Interpolated Melting Temperature (K)', fontweight='bold')
    plt.legend()
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
        
        # Calculate average magnitude/inte
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
    # eutectic_fig()
    # inter_figure()
    # inter_figure_raw()
    # inter_figure_filtered()
    inter_figure_correction()
    # inter_figure_error_metrics()
    
    # # Run hull metrics analysis
    # ternary_hull_metrics(include_predicted=True)
    
    # ternary_hull_metrics(include_predicted=False)
    
    # plot_L_parameter_distributions()
    
    # Run binary L parameter analysis with different deviation metrics
    # metrics = ['mad', 'std', 'range', 'pairwise']
    # metrics = ['range']
    
    # for metric in metrics:
    #     print(f"\n{'='*50}")
    #     print(f"Binary L Parameter Analysis using {metric.upper()}")
    #     print(f"{'='*50}")
    #     print(f"\n=== All Systems ({metric.upper()}) ===")
    #     binary_L_parameter_analysis(include_predicted=True, deviation_metric=metric)
        

if __name__ == "__main__":
    main()
