from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd

dump_dir = "all_dumps/for_PC/"
read_dir = "all_dumps/binary_fits/"

# Set to a Celsius value (for example, 650.0) to generate a free-energy slice
# and highlight the corresponding isotherm. Set to None to disable both.
ENERGY_TEMP_C = 1350
# ENERGY_TEMP_C = 1700
SHOW_ENERGY_COMPOSITION_TRIANGLE = False

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def plot_ternary_system():
    # Bi-Cd-Sn system
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    # tern_sys = ["Cd", "Sn", "As"]
    # tern_sys = ["Er", "Cu", "Ge"]
    # tern_sys = ["Ce", "Zn", "In"]
    # tern_sys = ["Tm", "Cu", "Ge"]
    tern_sys = ["Fe", "Ce", "Si"]
    # tern_sys = ["Bi", "Cd", "Sn"]
    # tern_sys = ["Er", "Mn", "Ge"]
    # tern_sys = ["Ce", "Fe", "Si"]
    # tern_sys = ["Al", "Fe", "Ge"]
    # tern_sys = ["Sm", "Fe", "Ge"]
    # tern_sys = ["Ge", "Ti", "Bi"]
    # tern_sys = ["Ba", "Mg", "Si"]
    tern_param_format = 'combined'  
    # bin_param_format = 'linear'
    # tern_param_format = 'linear'
    spec_inter = "Sm(FeGe)2"
    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")
    binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")
    # binary_param_df = pd.read_excel("data/ternary_dft_data/linear_le_s5e-5_w3e-2_p3-filtered-matrix.xlsx")
    # binary_param_df = pd.read_excel("data/ternary_dft_data/combexp_le_s10e-5_w5e-3_p8.5-filtered-ml.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")


    sorted_sys = sorted(tern_sys)
    binary_sys_labels = [
        f"{sorted_sys[0]}-{sorted_sys[1]}",
        f"{sorted_sys[1]}-{sorted_sys[2]}",
        f"{sorted_sys[2]}-{sorted_sys[0]}"
    ]

    # print(binary_sys_labels)

    binary_L_dict = {}

    sorted_sys = sorted(tern_sys)

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

        # Extract parameters and flip L1 signs if order was changed
        L0_a = float(params["L0_a"])
        L0_b = float(params["L0_b"])
        L1_a = float(params["L1_a"])
        L1_b = float(params["L1_b"])
        
        if order_changed:
            # Flip L1 parameter signs when element order is reversed
            L1_a = -L1_a
            L1_b = -L1_b
        
        binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]



    l0_tern = 0.0
    # l0_tern = 53000
    # l0_tern = 40000
    # l0_tern = 100000

    # print(binary_L_dict)
    # plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
    #                               L_dict=binary_L_dict, temp_slider=[0, -250], T_incr=10, delta=0.025, fit_or_pred=fitorpred)

    # print(fitorpred)

    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
                                  L_dict=binary_L_dict, temp_slider=[0, 0], T_incr=10, delta=0.01, fit_or_pred=fitorpred, L_tern = [l0_tern, 0])
    plotter.interpolate()
    # print(plotter.hsx_df)

    # manual adjustment of solid phase entropies
    # s_zrte = -1.57
    # plotter.hsx_df.loc[plotter.hsx_df['Phase Name'] == 'ZrTe', 'S'] = s_zrte
    # print(plotter.hsx_df)

    plotter.process_data()

    energy_result = None
    if ENERGY_TEMP_C is not None:
        try:
            energy_result = plotter.plot_free_energy_slice(
                ENERGY_TEMP_C, show_composition_triangle=SHOW_ENERGY_COMPOSITION_TRIANGLE
            )
            temp_tag = f"{energy_result['temperature_c']:.2f}".replace('-', 'm').replace('.', 'p')
            energy_filename = dump_dir + f'{"-".join(sorted_sys)}_energy_{temp_tag}C.html'
            ploff.plot(energy_result['figure'], filename=energy_filename, auto_open=True)
            print(
                f"Requested energy slice at {energy_result['requested_temperature_c']:.2f} C; "
                f"using nearest grid temperature {energy_result['temperature_c']:.2f} C."
            )
            print(f"Saved free-energy slice to: {energy_filename}")
        except ValueError as exc:
            print(f"Free-energy slice error: {exc}")
            c_grid = plotter.T_grid - 273.15
            print(
                f"Available grid temperatures in C are from {c_grid.min():.2f} to {c_grid.max():.2f} "
                f"with nominal increment {plotter.T_incr:.2f}."
            )

    tern_fig = plotter.plot_ternary()
    if energy_result is not None:
        try:
            plotter.add_temperature_isoline(tern_fig, energy_result['temperature_c'])
        except ValueError as exc:
            print(f"Temperature-isoline error: {exc}")

    # print(plotter.liq_plotting_df)
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

    # print(plotter.equil_df_list)

    # exctract melting temperatures of specific phases
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


if __name__ == "__main__":
    plot_ternary_system()
