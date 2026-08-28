from gliquid.binary import BinaryLiquid, BLPlotter
import pandas as pd

binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")

binary_sys = "Cu-Mg"
param_format = "combined"
pd_ind = 0
param_columns = ['L0_a', 'L0_b', 'L1_a', 'L1_b']
param_row = binary_param_df.loc[binary_param_df['system'].eq(binary_sys), param_columns]
if len(param_row) != 1:
    raise ValueError(f"Expected one fitted parameter row for {binary_sys}, found {len(param_row)}.")

params = param_row.iloc[0].astype(float).tolist()
binsys = BinaryLiquid.from_cache(binary_sys, params=params, param_format=param_format, pd_ind=pd_ind)
binplotter = BLPlotter(binsys)
liquidus_fig = binplotter.get_plot('fit+liq')

hsx_fig = binsys.hsx.plot_hsx()
hsx_fig.show()
