from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast

dump_dir = "all_dumps/gliq_manu_test_eut_ultimate/"
read_dir = "all_dumps/binary_fits/"

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def ternary_to_cartesian(coord):
    # Transformation matrix used in the original function
    unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    
    # Invert the transformation matrix
    inv_unitvec = np.linalg.inv(unitvec)
    
    # Apply the inverse transformation
    cart_coord = np.dot(coord, inv_unitvec)
    
    return cart_coord

def main():
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"

    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_comb_exp_with_soft_LE_10_opts-merged-hard_filtered-60elt-matrix.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")

    ternary_sys_list = [
        ['Bi', 'Ga', 'Sn'],
        ['Bi', 'Cd', 'Sn'],
        ['Ga', 'Sn', 'Zn'],
        ['Cd', 'Sn', 'Tl'],
        ['Cd', 'Sn', 'Zn'],
        ['Bi', 'Pb', 'Sn'],
        # ['Ag', 'Bi', 'Pb'],
        # ['Ag', 'Ni', 'Pb'],
        # ['Cr', 'Cu', 'Ru'],
        # ['Fe', 'La', 'Pu'],
        # ['Hf', 'Th', 'V'],
        # ['Cr', 'Sc', 'U']
    ]

    
    eut_temps = []
    eut_comps = []
    
    for tern_sys in ternary_sys_list:
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

            L0_a = float(params["L0_a"])
            L0_b = float(params["L0_b"])
            L1_a = float(params["L1_a"])
            L1_b = float(params["L1_b"])

            if order_changed:
                # Flip L1 parameter signs when element order is reversed
                L1_a = -L1_a
                L1_b = -L1_b

            binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]


        print(binary_L_dict)
        plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type=interp, param_format=tern_param_format,
                                    L_dict=binary_L_dict, temp_slider=[0, 0], T_incr=5.0, delta=0.01, fit_or_pred=fitorpred)

        plotter.interpolate()
        plotter.process_data()

        # bin_fig_list = plotter.bin_fig_list
        # for i, binary_plot in enumerate(bin_fig_list):
        #     ploff.plot(binary_plot, filename=dump_dir + f'{"-".join(sorted_sys)}_{interp}_binary_{i+1}.html', auto_open=True)

        df_list = plotter.equil_df_list

        # Find the lowest temperature ternary eutectic (where all three components > 0)
        for i in range(len(df_list)):
            if df_list[i]['Phase'].iloc[0] == 'L':
                sub_df = (df_list[i][df_list[i]['Phase'] == 'L'])
                x0_avg = sub_df['x0'].mean()
                x1_avg = sub_df['x1'].mean()
                x2_avg = 1 - x0_avg - x1_avg
                
                # Check if this is a true ternary composition (none of the components are zero)
                # Using a small tolerance to account for numerical precision
                tolerance = 1e-6
                if x0_avg > tolerance and x1_avg > tolerance and x2_avg > tolerance:
                    eut_temps.append(df_list[i]['T'].iloc[0])
                    comp = [x0_avg, x1_avg]
                    comp = ternary_to_cartesian(comp)
                    eut_comps.append(comp)
                    break   

        tern_fig = plotter.plot_ternary()
        sys_name = "-".join(sorted_sys)
        ploff.plot(tern_fig, filename=dump_dir + f'{sys_name}_{interp}_eutectic.html', auto_open=False)

    final_df = pd.DataFrame(list(zip(ternary_sys_list, eut_temps, eut_comps)), columns=['Ternary', 'Eutectic Temperature', 'Eutectic Composition'])

    # save to a new excel file
    final_df.to_excel(os.path.join(dump_dir, 'ternary_eutectic_results.xlsx'), index=False)


if __name__ == "__main__":
    main()