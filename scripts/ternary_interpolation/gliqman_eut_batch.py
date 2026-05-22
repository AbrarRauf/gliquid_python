from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
import traceback

dump_dir = "all_dumps/gliq_manu_eut_new/"
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

    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    # binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")
    # binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")


    ternary_sys_list = [
        ['Bi', 'Ga', 'Sn'],
        ['Bi', 'Cd', 'Sn'],
        ['Ga', 'Sn', 'Zn'],
        ['Cd', 'Sn', 'Tl'],
        ['Cd', 'Sn', 'Zn'],
        ['Bi', 'Pb', 'Sn'],
        ['Al', 'Be', 'Si'],
        ['Al', 'Ga', 'Ge'],
        ['Al', 'Ga', 'Zn'],
        ['Al', 'Si', 'Zn'],
        ['Au', 'Ge', 'Tl'],
        ['Bi', 'Cd', 'Ge'],
        ['Bi', 'Cd', 'Pb'],
        ['Bi', 'Cd', 'Sn'],
        ['Bi', 'Ge', 'Pb'],
        ['Bi', 'Ge', 'Sn'],
        ['Bi', 'Pb', 'Sn'],
        ['Cd', 'Ge', 'Pb'],
        ['Cd', 'Ge', 'Sn'],
        ['Cd', 'Ge', 'Tl'],
        ['Cd', 'Pb', 'Sn'],
        ['Cd', 'Si', 'Zn'],
        ['Cd', 'Sn', 'Zn'],
        ['Ga', 'Ge', 'In'],
        ['Ga', 'Ge', 'Sn'],
        ['Ga', 'Sn', 'Zn'],
        ['Ge', 'Pb', 'Sn'],
        ['Hf', 'Mo', 'Th'],
        ['Hf', 'Mo', 'Y'],
    ]

    # only keep unique systems (e.g. remove duplicates like ['Bi', 'Cd', 'Sn'] and ['Bi', 'Sn', 'Cd'])
    unique_ternary_sys_list = []
    for sys in ternary_sys_list:
        sorted_sys = sorted(sys)
        if sorted_sys not in unique_ternary_sys_list:
            unique_ternary_sys_list.append(sorted_sys)

    ternary_sys_list = unique_ternary_sys_list
    print(len(ternary_sys_list))
    
    eut_temps = []
    eut_comps = []
    successful_systems = []
    failure_log = []
    
    for tern_sys in ternary_sys_list:
        sorted_sys = sorted(tern_sys)
        try:
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
                # elif bin_sys in binary_param_pred_df['system'].tolist():
                #     params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                #     fitorpred[bin_sys] = "pred"
                # elif flipped_sys in binary_param_pred_df['system'].tolist():
                #     params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                #     fitorpred[bin_sys] = "pred"
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

            bin_fig_list = plotter.bin_fig_list
            for i, binary_plot in enumerate(bin_fig_list):
                ploff.plot(binary_plot, filename=dump_dir + f'{"-".join(sorted_sys)}_{interp}_binary_{i+1}.html', auto_open=False)

            df_list = plotter.equil_df_list

            # Find the lowest temperature ternary eutectic (where all three components > 0)
            found_eutectic = False
            # for i in range(len(df_list)):
            #     if df_list[i]['Phase'].iloc[0] == 'L':
            #         sub_df = (df_list[i][df_list[i]['Phase'] == 'L'])
            #         x0_avg = sub_df['x0'].mean()
            #         x1_avg = sub_df['x1'].mean()
            #         x2_avg = 1 - x0_avg - x1_avg

            #         # Check if this is a true ternary composition (none of the components are zero)
            #         # Using a small tolerance to account for numerical precision
            #         tolerance = 1e-6
            #         if x0_avg > tolerance and x1_avg > tolerance and x2_avg > tolerance:
            #             eut_temps.append(df_list[i]['T'].iloc[0])
            #             comp = [x0_avg, x1_avg]
            #             comp = ternary_to_cartesian(comp)
            #             eut_comps.append(comp)
            #             successful_systems.append(tern_sys)
            #             found_eutectic = True
            #             break

            # if not found_eutectic:
            #     raise ValueError("No ternary eutectic liquid point found in equilibrium slices.")

            for i in range(len(df_list)):
                if df_list[i]['Phase'].iloc[0] == 'L':
                    eut_temps.append(df_list[i]['T'].iloc[0])
                    sub_df = (df_list[i][df_list[i]['Phase'] == 'L'])
                    x0_avg = sub_df['x0'].mean()
                    x1_avg = sub_df['x1'].mean()
                    comp = [x0_avg, x1_avg]
                    comp = ternary_to_cartesian(comp)
                    eut_comps.append(comp)
                    successful_systems.append(tern_sys)
                    found_eutectic = True
                    break

            if not found_eutectic:
                raise ValueError("No liquid point found in equilibrium slices.")

            tern_fig = plotter.plot_ternary()
            sys_name = "-".join(sorted_sys)
            ploff.plot(tern_fig, filename=dump_dir + f'{sys_name}_{interp}_eutectic.html', auto_open=False)

        except Exception as e:
            failure_log.append(
                {
                    "system": tern_sys,
                    "sorted_system": sorted_sys,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Failed system {'-'.join(sorted_sys)}: {e}")
            continue

    final_df = pd.DataFrame(
        list(zip(successful_systems, eut_temps, eut_comps)),
        columns=['Ternary', 'Eutectic Temperature', 'Eutectic Composition']
    )

    # save to a new excel file
    final_df.to_excel(os.path.join(dump_dir, 'ternary_eutectic_results.xlsx'), index=False)

    with open(os.path.join(dump_dir, 'ternary_eutectic_failures.json'), 'w') as f:
        json.dump(failure_log, f, indent=4)


if __name__ == "__main__":
    main()