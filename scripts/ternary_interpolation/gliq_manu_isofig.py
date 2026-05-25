import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np 
import os
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


dump_dir = "all_dumps/test_main/"
# df = pd.read_csv("ternary_gtx_test.csv")
# df = pd.read_csv(dump_dir + "ternary_gtx_test2.csv")
df = pd.read_csv(dump_dir + "ternary_gtx_test3.csv")


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

# Create the main figure with the filled contour plot
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
    ),
    colorbar = dict(
        title = dict(text='Temperature (K)'),
        orientation = 'v',
        thickness = 24,
        len = 0.85,
        x = 1.03,
        y = 0.5
    )
))

# Add white isothermal contour lines below the main plot
# Calculate appropriate contour levels
temp_range = liquid_df['T'].max() - liquid_df['T'].min()
contour_levels = np.linspace(liquid_df['T'].min(), liquid_df['T'].max(), 10)

fig.add_trace(go.Contour(
    x = xi[0], 
    y = yi[:, 0],
    z = zi,
    showscale = False,
    colorscale = [[0, 'white'], [1, 'white']],  # Force white colorscale
    line = dict(
        color = 'white',
        width = 2),
    contours = dict(
        coloring = 'lines',
        showlines = True,
        start = liquid_df['T'].min(),
        end = liquid_df['T'].max(),
        size = temp_range / 19,  # 9 intervals for 10 contour lines
    ),
    hoverinfo = 'skip',  # Disable hover for contour lines
    name = ''  # Empty name to avoid legend entry
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
        cmin=liquid_df['T'].min(), 
        cmax=liquid_df['T'].max(),
        showscale=False
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


# Save the main interactive figure as HTML with a vertical temperature colorbar.
main_html_path = os.path.join(dump_dir, "gliq_manu_isofig_main.html")
fig.write_html(main_html_path, include_plotlyjs='cdn')

# Create a standalone horizontal Matplotlib colorbar for manuscript use.
MANUSCRIPT_TICK_FONTSIZE = 18
MANUSCRIPT_LABEL_FONTSIZE = 20

cb_fig, cb_ax = plt.subplots(figsize=(9, 1.6))
norm = Normalize(vmin=float(liquid_df['T'].min()), vmax=float(liquid_df['T'].max()))
sm = ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])

cbar = cb_fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
cbar.set_label('Temperature (K)', fontsize=MANUSCRIPT_LABEL_FONTSIZE, fontweight='bold', labelpad=8)
cbar.ax.tick_params(labelsize=MANUSCRIPT_TICK_FONTSIZE, width=1.5, length=6)
for tick_label in cbar.ax.get_xticklabels():
    tick_label.set_fontweight('bold')

cb_fig.tight_layout()
hbar_png_path = os.path.join(dump_dir, "gliq_manu_isofig_horizontal_colorbar.png")
hbar_svg_path = os.path.join(dump_dir, "gliq_manu_isofig_horizontal_colorbar.svg")
cb_fig.savefig(hbar_png_path, dpi=600, bbox_inches='tight')
cb_fig.savefig(hbar_svg_path, bbox_inches='tight')
plt.close(cb_fig)

print(f"Saved main HTML figure: {main_html_path}")
print(f"Saved horizontal colorbar PNG: {hbar_png_path}")
print(f"Saved horizontal colorbar SVG: {hbar_svg_path}")


fig.show()