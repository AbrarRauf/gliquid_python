from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd

dump_dir = "all_dumps/for_PC/"
read_dir = "all_dumps/binary_fits/"

# and highlight the corresponding isotherm. Set to None to disable both.
ENERGY_TEMP_C = 1350
# ENERGY_TEMP_C = 1700
SHOW_ENERGY_COMPOSITION_TRIANGLE = True
SHOW_ENERGY_SOLID_LABELS = False
CLIP_LIQUID_TO_LOWER_HULL = True
SHOW_TERNARY_AXES = False
SHOW_TERMINAL_REFERENCE_LABELS = False
# Places tern_sys[0] at the apex and tern_sys[2]-tern_sys[1] along the base.
GENERATE_LIQUIDUS_PROJECTION = False
# Set to None to retain the generated binary order.
# BINARY_PLOT_ORDER = ["Si-Ce", "Fe-Ce", "Si-Fe"]
BINARY_PLOT_ORDER = None

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def _mirror_binary_figure(fig, left_component, right_component):
    for trace in fig.data:
        if trace.x is not None:
            trace.x = [None if value is None else 100 - float(value) for value in trace.x]

    for annotation in fig.layout.annotations or ():
        if annotation.x is not None:
            annotation.x = (1 if annotation.xref == 'paper' else 100) - float(annotation.x)
        if annotation.xanchor == 'left':
            annotation.xanchor = 'right'
        elif annotation.xanchor == 'right':
            annotation.xanchor = 'left'

    legend = fig.layout.legend
    if legend.x is not None:
        legend.x = 1 - float(legend.x)
    if legend.xanchor == 'left':
        legend.xanchor = 'right'
    elif legend.xanchor == 'right':
        legend.xanchor = 'left'
    fig.update_layout(title_text=f'<b>{left_component}-{right_component} DFT-Referenced Phase Diagram</b>')


def _order_binary_figures(figures, generated_systems, requested_order=None):
    if requested_order is None:
        return figures
    if len(requested_order) != len(figures):
        raise ValueError(f"Expected {len(figures)} binary systems, received {len(requested_order)}.")

    figure_map = {}
    for system, figure in zip(generated_systems, figures):
        components = system.split('-')
        figure_map[frozenset(components)] = (figure, tuple(sorted(components)))

    ordered_figures = []
    seen = set()
    for system in requested_order:
        components = system.split('-')
        if len(components) != 2 or components[0] == components[1]:
            raise ValueError(f"Invalid binary system '{system}'. Use the form 'A-B'.")
        key = frozenset(components)
        if key not in figure_map:
            raise ValueError(f"Requested binary system '{system}' is not available.")
        if key in seen:
            raise ValueError(f"Binary system '{system}' is listed more than once.")
        seen.add(key)

        figure, canonical_order = figure_map[key]
        if tuple(components) != canonical_order:
            _mirror_binary_figure(figure, *components)
        else:
            figure.update_layout(title_text=f'<b>{system} DFT-Referenced Phase Diagram</b>')
        ordered_figures.append(figure)

    return ordered_figures


def plot_ternary_system(binary_plot_order=None):
    # Bi-Cd-Sn system
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    # tern_sys = ["Cd", "Sn", "As"]
    # tern_sys = ["Er", "Cu", "Ge"]
    # tern_sys = ["Ce", "Zn", "In"]
    # tern_sys = ["Tm", "Cu", "Ge"]
    # tern_sys = ["Fe", "Ce", "Si"]
    tern_sys = ["Ce", "Zn", "In"]
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
                ENERGY_TEMP_C,
                show_composition_triangle=SHOW_ENERGY_COMPOSITION_TRIANGLE,
                show_solid_labels=SHOW_ENERGY_SOLID_LABELS,
                show_terminal_labels=SHOW_TERMINAL_REFERENCE_LABELS,
                clip_liquid_to_lower_hull=CLIP_LIQUID_TO_LOWER_HULL,
            )
            temp_tag = f"{energy_result['temperature_c']:.2f}".replace('-', 'm').replace('.', 'p')
            energy_filename = dump_dir + f'{"-".join(sorted_sys)}_energy_{temp_tag}C.html'
            colorbar_filename = dump_dir + f'{"-".join(sorted_sys)}_energy_{temp_tag}C_colorbar.png'
            vertical_colorbar_filename = dump_dir + f'{"-".join(sorted_sys)}_energy_{temp_tag}C_colorbar_vertical.png'
            left_colorbar_filename = dump_dir + f'{"-".join(sorted_sys)}_energy_{temp_tag}C_colorbar_vertical_left.png'
            ploff.plot(energy_result['figure'], filename=energy_filename, auto_open=True)
            # exit()
            plotter.save_free_energy_colorbar(energy_result, colorbar_filename)
            plotter.save_free_energy_colorbar(energy_result, vertical_colorbar_filename, orientation='vertical')
            plotter.save_free_energy_colorbar(
                energy_result, left_colorbar_filename, orientation='vertical', label_side='left'
            )
            print(
                f"Requested energy slice at {energy_result['requested_temperature_c']:.2f} C; "
                f"using nearest grid temperature {energy_result['temperature_c']:.2f} C."
            )
            print(f"Saved free-energy slice to: {energy_filename}")
            print(f"Saved free-energy colorbar to: {colorbar_filename}")
            print(f"Saved vertical free-energy colorbar to: {vertical_colorbar_filename}")
            print(f"Saved left-side free-energy colorbar to: {left_colorbar_filename}")
        except ValueError as exc:
            print(f"Free-energy slice error: {exc}")
            c_grid = plotter.T_grid - 273.15
            print(
                f"Available grid temperatures in C are from {c_grid.min():.2f} to {c_grid.max():.2f} "
                f"with nominal increment {plotter.T_incr:.2f}."
            )

    tern_fig = plotter.plot_ternary(
        show_axes=SHOW_TERNARY_AXES,
        show_terminal_labels=SHOW_TERMINAL_REFERENCE_LABELS,
    )
    if energy_result is not None:
        try:
            plotter.add_temperature_isoline(tern_fig, energy_result['temperature_c'])
        except ValueError as exc:
            print(f"Temperature-isoline error: {exc}")

    if GENERATE_LIQUIDUS_PROJECTION:
        try:
            projection_result = plotter.plot_liquidus_projection(apex_component=tern_sys[0])
            projection_filename = dump_dir + f'{"-".join(sorted_sys)}_liquidus_projection.html'
            projection_colorbar_filename = dump_dir + f'{"-".join(sorted_sys)}_liquidus_projection_colorbar.png'
            ploff.plot(projection_result['figure'], filename=projection_filename, auto_open=True)
            plotter.save_liquidus_projection_colorbar(projection_result, projection_colorbar_filename)
            base_left, base_right = projection_result['base_components']
            print(
                f"Saved liquidus projection with {base_left}-{base_right} at the base and "
                f"{projection_result['apex_component']} at the apex to: {projection_filename}"
            )
            print(f"Saved liquidus projection colorbar to: {projection_colorbar_filename}")
        except ValueError as exc:
            print(f"Liquidus-projection error: {exc}")

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

    bin_fig_list = _order_binary_figures(
        plotter.bin_fig_list, list(plotter.L_dict), binary_plot_order
    )
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
    plot_ternary_system(binary_plot_order=BINARY_PLOT_ORDER)
