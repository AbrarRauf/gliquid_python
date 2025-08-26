from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd

dump_dir = "all_dumps/zrte_spec/"
read_dir = "all_dumps/binary_fits/"

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def plot_ternary_system():
    # Bi-Cd-Sn system
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_sys = ["Nd", "Fe", "B"]
    tern_param_format = 'combined'
    # bin_param_format = 'linear'
    # tern_param_format = 'linear'
    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")
    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/v17_comb1S_tau10k_predictions_rf_optimized.xlsx")



    sorted_sys = sorted(tern_sys)
    binary_sys_labels = [
        f"{sorted_sys[0]}-{sorted_sys[1]}",
        f"{sorted_sys[1]}-{sorted_sys[2]}",
        f"{sorted_sys[2]}-{sorted_sys[0]}"
    ]

    print(binary_sys_labels)

    binary_L_dict = {}

    sorted_sys = sorted(tern_sys)

    fitorpred = {}
    for bin_sys in binary_sys_labels:
        flipped_sys = "-".join(sorted(bin_sys.split('-')))

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

        binary_L_dict[bin_sys] = [
            float(params["L0_a"]),
            float(params["L0_b"]),
            float(params["L1_a"]),
            float(params["L1_b"])
        ]

    # for sys in binary_sys_labels:
    #     flipped_sys = "-".join(sorted(sys.split("-")))
    #     print(flipped_sys)
    #     json_path = os.path.join(read_dir, f"{sys}_{bin_param_format}_fit_results.json")
    #     if not os.path.exists(json_path):
    #         json_path = os.path.join(read_dir, f"{flipped_sys}_{bin_param_format}_fit_results.json")
            
    #     with open(json_path, "r") as f:
    #         params = json.load(f)
    #     binary_L_dict[sys] = [
    #         params["L0_a"],
    #         params["L0_b"],
    #         params["L1_a"],
    #         params["L1_b"]
    #     ]

    print(binary_L_dict)

    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
                                  L_dict=binary_L_dict, temp_slider=[0, 1000], T_incr=5, delta=0.01, fit_or_pred=fitorpred)

    plotter.interpolate()
    plotter.process_data()
    tern_fig = plotter.plot_ternary()
    # ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_{tern_param_format}_system.html', auto_open=True)
    ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_trial_system.html', auto_open=True)


if __name__ == "__main__":
    plot_ternary_system()
