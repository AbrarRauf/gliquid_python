import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np 
from scipy.interpolate import griddata


dump_dir = "all_dumps/zrte_spec/"
# df = pd.read_csv("ternary_gtx_test.csv")
df = pd.read_csv(dump_dir + "ternary_gtx_test.csv")


print(df)


def ternary_to_cartesian(coord):
    # Transformation matrix used in the original function
    unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    
    # Invert the transformation matrix
    inv_unitvec = np.linalg.inv(unitvec)
    
    # Apply the inverse transformation
    cart_coord = np.dot(coord, inv_unitvec)
    
    return cart_coord

df['x0'] = df['x0'].astype(float)
df['x1'] = df['x1'].astype(float)

# convert the ternary coordinates to cartesian coordinates
df['cartesian_coords'] = df[['x0', 'x1']].apply(lambda x: ternary_to_cartesian(x), axis=1)

# convert T by adding 273.15 to convert from Celsius to Kelvin
df['T'] = df['T'].astype(float) + 273.15

solid_df = df[df['Phase'] != 'L']
liquid_df = df[df['Phase'] == 'L']
solid_df = solid_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='last')
liquid_df = liquid_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='first')

print(liquid_df)
print(solid_df)


xi = np.linspace(liquid_df['x0'].min(), liquid_df['x0'].max(), 200)
yi = np.linspace(liquid_df['x1'].min(), liquid_df['x1'].max(), 200)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((liquid_df['x0'], liquid_df['x1']), liquid_df['T'], (xi, yi), method='linear')

fig = go.Figure(go.Contour(
    x = xi[0], 
    y = yi[:, 0],
    z = zi,
    colorscale = 'viridis',
    zmin = liquid_df['T'].min(),
    zmax = liquid_df['T'].max(),
    showscale = True,
    line = dict(
        width = 0.0),
    contours = dict(
        coloring = 'heatmap',
        showlines = False,
    )
))

# Add the scatter plot for solid phases
fig.add_trace(go.Scattergl(
    x=solid_df['x0'],
    y=solid_df['x1'],
    mode='markers',
    marker=dict(
        size=15,
        color=solid_df['T'],
        colorscale='viridis', # Use the same colorscale as contour
        line=dict(width=2, color='black'), # Black border
        cmin=liquid_df['T'].min(), # Ensure consistent coloring with contour
        cmax=liquid_df['T'].max()  # Ensure consistent coloring with contour
    ),
    text=solid_df['Phase'],
    hoverinfo='x+y+text',
    showlegend=False,
))


fig.add_trace(
    go.Scatter(
        x = [0, 0.5/1, 1/1, 0],
        y = [0, np.sqrt(3)/2, 0, 0],
        mode = 'lines',
        line = dict(color='black', width=3.5),
        showlegend = False,
    )
)


fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)

# update figure width and height
wd = ht = 1600

fig.update_layout(
    width=1200,
    height=1000,
)


fig.show()