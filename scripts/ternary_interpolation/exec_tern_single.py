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
    # tern_sys = ["Cd", "Sn", "As"]
    # tern_sys = ["Zr", "Te", "Bi"]
    # tern_sys = ["Ge", "Ti", "Bi"]
    tern_sys = ["Fe", "Ce", "Si"]
    tern_param_format = 'combined'
    # bin_param_format = 'linear'
    # tern_param_format = 'linear'
    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")
    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")


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



    l0_tern = 0.0
    # l0_tern = 53000
    # l0_tern = -70000

    print(binary_L_dict)
    # plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
    #                               L_dict=binary_L_dict, temp_slider=[0, -250], T_incr=10, delta=0.025, fit_or_pred=fitorpred)

    print(fitorpred)

    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
                                  L_dict=binary_L_dict, temp_slider=[0, 0], T_incr=10, delta=0.025, fit_or_pred=fitorpred, L_tern = [l0_tern, 0])
    plotter.interpolate()
    print(plotter.hsx_df)

    # manual adjustment of solid phase entropies
    # s_zrte = -1.57
    # plotter.hsx_df.loc[plotter.hsx_df['Phase Name'] == 'ZrTe', 'S'] = s_zrte
    # print(plotter.hsx_df)

    plotter.process_data()

    tern_fig = plotter.plot_ternary()

    # update layout and remove axis and background
    # tern_fig.update_layout(
    #     scene = dict(
    #         zaxis_visible=False,
    #         xaxis_visible=False,
    #         yaxis_visible=False,
    #         bgcolor='white'
    #     )
    # )
    bin_fig_list = plotter.bin_fig_list
    for i, bin_fig in enumerate(bin_fig_list):
        bin_fig.show()

    print(plotter.equil_df_list)

    # exctract melting temperatures of specific phases
    # spec_inter = "ZrTe"
    # inter_list = [spec_inter]
    # melting_temps = plotter.get_inter_melting_temps(inter_list)
    # print(melting_temps)
    # print("For l0_tern =", l0_tern, "Melting point", melting_temps[spec_inter] + 273.15, "K")
    # print("For l0_tern =", l0_tern, "Melting point", melting_temps[spec_inter], "C")

    # order by index
    plotter.plotting_df = plotter.plotting_df.sort_index().reset_index(drop=True)
    # extract the first 5 named columns to a csv called ternary_gtx_test.csv in dump_dir
    plotter.plotting_df.iloc[:, :5].to_csv(dump_dir + "ternary_gtx_test3.csv", index=False)
    # ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_{tern_param_format}_system.html', auto_open=True)
    ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_eut_trial_system.html', auto_open=True)

    # print("For l0_tern =", l0_tern, "Melting point", melting_temps["CdSnAs2"] + 273.15, "K")

if __name__ == "__main__":
    plot_ternary_system()