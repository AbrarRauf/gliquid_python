from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
from collections import Counter

dump_dir = "all_dumps/zrte_x_ternaries/"

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)


def _param_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        lookup.setdefault(str(row["system"]), row)
    return lookup


def _get_binary_params(bin_sys, fitted_lookup, pred_lookup):
    reversed_sys = "-".join(reversed(bin_sys.split("-")))
    source = "fit"

    if bin_sys in fitted_lookup:
        row = fitted_lookup[bin_sys]
        param_sys = bin_sys
    elif reversed_sys in fitted_lookup:
        row = fitted_lookup[reversed_sys]
        param_sys = reversed_sys
    elif bin_sys in pred_lookup:
        row = pred_lookup[bin_sys]
        param_sys = bin_sys
        source = "pred"
    elif reversed_sys in pred_lookup:
        row = pred_lookup[reversed_sys]
        param_sys = reversed_sys
        source = "pred"
    else:
        raise ValueError(f"Binary system {bin_sys} not found in fitted or predicted parameter dataframes.")

    L0_a = float(row["L0_a"])
    L0_b = float(row["L0_b"])
    L1_a = float(row["L1_a"])
    L1_b = float(row["L1_b"])

    if param_sys != bin_sys:
        L1_a = -L1_a
        L1_b = -L1_b

    return [L0_a, L0_b, L1_a, L1_b], source, param_sys


def main():
    only_include = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Eu',
                    'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb',
                    'Nd', 'Ni', 'Os', 'Pb', 'Pr', 'Rb', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Th',
                    'Ti', 'Tl', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
    x_elements = [el for el in only_include if el not in {"Zr", "Te"}]

    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "kohler"

    binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")

    required_cols = {"system", "L0_a", "L0_b", "L1_a", "L1_b"}
    missing_fitted = required_cols - set(binary_param_df.columns)
    missing_pred = required_cols - set(binary_param_pred_df.columns)
    if missing_fitted:
        raise ValueError(f"binary_param_df is missing required columns: {sorted(missing_fitted)}")
    if missing_pred:
        raise ValueError(f"binary_param_pred_df is missing required columns: {sorted(missing_pred)}")

    fitted_lookup = _param_lookup(binary_param_df)
    pred_lookup = _param_lookup(binary_param_pred_df)
    meta_data = {}
    Error_dict = {}

    for i, x_el in enumerate(x_elements):
        tern_sys = ["Zr", "Te", x_el]
        sorted_sys = sorted(tern_sys)
        system_label = "-".join(sorted_sys)
        print(f"System {system_label} with index {i}")

        try:
            binary_sys_labels = [
                f"{sorted_sys[0]}-{sorted_sys[1]}",
                f"{sorted_sys[1]}-{sorted_sys[2]}",
                f"{sorted_sys[2]}-{sorted_sys[0]}",
            ]

            binary_L_dict = {}
            fitorpred = {}
            param_systems = {}
            predicted_binaries = []

            for bin_sys in binary_sys_labels:
                params, source, param_sys = _get_binary_params(bin_sys, fitted_lookup, pred_lookup)
                binary_L_dict[bin_sys] = params
                fitorpred[bin_sys] = source
                param_systems[bin_sys] = param_sys
                if source == "pred":
                    predicted_binaries.append(bin_sys)

            if predicted_binaries:
                print(f"Using predicted params for {system_label}: {predicted_binaries}")

            plotter = ternary_gtx_plotter(
                tern_sys, data_dir, interp_type=interp, param_format=tern_param_format,
                L_dict=binary_L_dict, temp_slider=[0, 500], T_incr=5.0, delta=0.025,
                fit_or_pred=fitorpred,
            )
            plotter.interpolate()
            plotter.process_data()
            tern_fig = plotter.plot_ternary()
            html_path = os.path.join(dump_dir, f"{system_label}_{interp}_system.html")
            ploff.plot(tern_fig, filename=html_path, auto_open=False)

            meta_data[system_label] = {
                "system": system_label,
                "index": int(i),
                "x_element": x_el,
                "Fit Type": "Contains predicted" if predicted_binaries else "All fitted",
                "predicted_binaries": predicted_binaries,
                "fit_or_pred": fitorpred,
                "param_systems": param_systems,
                "ternary_meta": plotter.ternary_meta,
                "binary_L_params": binary_L_dict,
                "html_file": html_path,
            }
            print(f"Saved {html_path}")

        except Exception as e:
            print(f"Error in system {system_label} with index {i}: {e}")
            Error_dict[system_label] = {
                "status": "error",
                "system": system_label,
                "index": int(i),
                "x_element": x_el,
                "message": str(e),
            }

    status_counts = Counter(v.get("status", "unknown") for v in Error_dict.values())
    print(f"Generated {len(meta_data)} systems")
    print(f"Error status counts: {dict(status_counts)}")

    with open(os.path.join(dump_dir, f"zrte_x_ternary_meta_{interp}.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    with open(os.path.join(dump_dir, f"zrte_x_ternary_errors_{interp}.json"), "w") as f:
        json.dump(Error_dict, f, indent=4)


if __name__ == "__main__":
    main()
