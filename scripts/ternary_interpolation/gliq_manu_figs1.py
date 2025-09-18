import pandas as pd 
import matplotlib.pyplot as plt
import os 
import ast
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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


def main():
    eutectic_fig()
    

if __name__ == "__main__":
    main()