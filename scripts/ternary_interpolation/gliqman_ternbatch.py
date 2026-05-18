from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast
from collections import Counter

dump_dir = "all_dumps/gliq_manu_forreal/"
read_dir = "all_dumps/binary_fits/"
print(data_dir)

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def main():
    '''ERROR ANALYSIS'''
    # # read in error json file from dump_dir
    # with open(os.path.join(dump_dir, f"ternary_Gliq_errors_linear.json"), "r") as f:
    #     Error_dict = json.load(f)


    # # initialize a dataframe
    # error_df = pd.DataFrame.from_dict(Error_dict, orient="index", columns=["error_message"])
    # error_df.index.name = "system"
    # error_df.reset_index(inplace=True)

    # print(error_df)

    # # extract the unique error messages to a list
    # unique_errors = error_df["error_message"].unique().tolist()
    # print(unique_errors)

    # spec_err = unique_errors[1]

    # # extract these systems to a list
    # spec_err_systems = error_df[error_df["error_message"] == spec_err]["system"].tolist()
    # print(spec_err_systems)
    # print(len(spec_err_systems))

    only_include = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Eu', 
                    'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 
                    'Nd', 'Ni', 'Os', 'Pb', 'Pr', 'Rb', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Th', 
                    'Ti', 'Tl', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
    print(len(only_include))

    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "combined"
    interp = "linear"

    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_comb_exp_with_soft_LE_10_opts-merged-hard_filtered-60elt-matrix.xlsx")
    binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")
    # binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")
    binary_param_pred_df = pd.read_excel("data/ternary_dft_data/final_ml_params-internal.xlsx")
    ternary_df = pd.read_excel("data/ternary_dft_data/ternary_im_filtered_IQR.xlsx")

    required_ternary_cols = {"elements", "melting_point_k", "reduced_formula"}
    missing_ternary_cols = required_ternary_cols - set(ternary_df.columns)
    if missing_ternary_cols:
        raise ValueError(f"ternary_df is missing required columns: {sorted(missing_ternary_cols)}")

    required_binary_cols = {"system", "L0_a", "L0_b", "L1_a", "L1_b", "mae", "rmse"}
    missing_binary_cols = required_binary_cols - set(binary_param_df.columns)
    if missing_binary_cols:
        raise ValueError(f"binary_param_df is missing required columns: {sorted(missing_binary_cols)}")

    ternary_sys_list = ternary_df["elements"].tolist()
    print(len(ternary_sys_list))
    ternary_sys_list = [ast.literal_eval(e) if isinstance(e, str) else e for e in ternary_sys_list]
    print(len(ternary_sys_list))
    print(ternary_sys_list)

    # print duplicate systems in ternary_sys_list
    system_counts = Counter(tuple(sorted(e)) for e in ternary_sys_list)
    duplicates = {system: count for system, count in system_counts.items() if count > 1}
    print(f"Duplicate systems and their counts: {duplicates}")
    
    system_list = binary_param_df["system"].tolist()
    
    congruent_temps = []
    types = []
    valid_idx = []

    meta_data = {}
    Error_dict = {}

    for i, tern_sys in enumerate(ternary_sys_list):
        entry_key = f"row_{i}"
        sorted_sys = None

        # tern_sys = ["Ba", "Mg", "Si"]
        # tern_sys = ["Ce", "Fe", "Si"]
        print(f"System {tern_sys} with index {i}")
        congruent_temp = ternary_df.iloc[i]["melting_point_k"]
        congruent_phase = ternary_df.iloc[i]["reduced_formula"]

        if not isinstance(tern_sys, (list, tuple)):
            Error_dict[entry_key] = {
                "status": "invalid_input",
                "system": str(tern_sys),
                "index": int(i),
                "reason": "elements field is not a list/tuple",
            }
            continue

        if len(tern_sys) != 3:
            Error_dict[entry_key] = {
                "status": "invalid_input",
                "system": str(tern_sys),
                "index": int(i),
                "reason": f"expected ternary system of length 3, got {len(tern_sys)}",
            }
            continue

        if len(set(tern_sys)) != 3:
            Error_dict[entry_key] = {
                "status": "invalid_input",
                "system": str(tern_sys),
                "index": int(i),
                "reason": "ternary system contains duplicate elements",
            }
            continue

        if pd.isna(congruent_phase) or str(congruent_phase).strip() == "":
            Error_dict[entry_key] = {
                "status": "invalid_input",
                "system": "-".join(sorted([str(e) for e in tern_sys])),
                "index": int(i),
                "reason": "missing reduced_formula in ternary_df",
            }
            continue

        sorted_sys = sorted([str(e) for e in tern_sys])
        entry_key = f"{'-'.join(sorted_sys)}__idx_{i}__phase_{congruent_phase}"

        if not all(elem in only_include for elem in sorted_sys):
            Error_dict[entry_key] = {
                "status": "skipped",
                "system": "-".join(sorted_sys),
                "index": int(i),
                "reason": "not_include_filter",
                "excluded_elements": [elem for elem in sorted_sys if elem not in only_include],
                "phase": str(congruent_phase),
            }
            continue

        try:
            binary_sys_labels = [
                f"{sorted_sys[0]}-{sorted_sys[1]}",
                f"{sorted_sys[1]}-{sorted_sys[2]}",
                f"{sorted_sys[2]}-{sorted_sys[0]}"
            ]
            print(binary_sys_labels)

            binary_L_dict = {}
            fitorpred = {}

            # for bin_sys in binary_sys_labels:
            #     flipped_sys = "-".join(sorted(bin_sys.split("-")))
            #     print(flipped_sys)

            #     if bin_sys in binary_param_df["system"].tolist():
            #         params = binary_param_df[binary_param_df["system"] == bin_sys].iloc[0]
            #     elif flipped_sys in binary_param_df["system"].tolist():
            #         params = binary_param_df[binary_param_df["system"] == flipped_sys].iloc[0]
            #     else:
            #         raise Exception("System not in df")

            #     binary_L_dict[bin_sys] = [
            #         float(params["L0_a"]),
            #         float(params["L0_b"]),
            #         float(params["L1_a"]),
            #         float(params["L1_b"])
            #     ]
            pred_tag = "All fitted"
            mae = []
            rmse = []
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                    mae.append(params["mae"])
                    rmse.append(params["rmse"])
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                    mae.append(params["mae"])
                    rmse.append(params["rmse"])
                else:
                    raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

                if pd.isna(params["mae"]) or pd.isna(params["rmse"]):
                    raise ValueError(f"Binary system {bin_sys} has missing mae/rmse metrics.")

                # elif bin_sys in binary_param_pred_df['system'].tolist():
                #     params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                #     fitorpred[bin_sys] = "pred"
                #     pred_tag = "Contains predicted"
                # elif flipped_sys in binary_param_pred_df['system'].tolist():
                #     params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                #     fitorpred[bin_sys] = "pred"
                #     pred_tag = "Contains predicted"
                # else:
                #     raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

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

            print(binary_L_dict)
            plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type=interp, param_format=tern_param_format,
                                        L_dict=binary_L_dict, temp_slider=[0, 500], T_incr=5.0, delta=0.025, fit_or_pred=fitorpred)
            plotter.interpolate()
            plotter.process_data()
            tern_meta = plotter.ternary_meta
            df_list = plotter.equil_df_list
            if len(df_list) == 0:
                raise ValueError("No equilibrium slices produced by process_data().")
            concat_df = pd.concat(df_list, ignore_index=True)
            sub_df = concat_df[concat_df["Phase"] == congruent_phase]
            tern_fig = plotter.plot_ternary()
            safe_phase = str(congruent_phase).replace('/', '_').replace(' ', '_')
            ploff.plot(
                tern_fig,
                filename=dump_dir + f'{"-".join(sorted_sys)}_{safe_phase}_idx{i}_{interp}_system.html',
                auto_open=False
            )
            if sub_df.empty:
                raise Exception("MPDS congruent phase not on the hull!")

            sub_df = sub_df.sort_values(by="T", ascending=False)
            sub_df = sub_df.iloc[0]
            comp = [sub_df["x0"], sub_df["x1"]]
            temp = sub_df["T"] + 273.15
            print(concat_df)
            sub_df2 = concat_df[(concat_df["Phase"] == "L") &
                                (np.isclose(concat_df["x0"], comp[0], rtol=0, atol=0.025)) &
                                (np.isclose(concat_df["x1"], comp[1], rtol=0, atol=0.025))]
            if sub_df2.empty:
                raise ValueError("No nearby liquid point found at congruent-phase composition.")
            sub_df2 = sub_df2.sort_values(by="T", ascending=True)
            sub_df2 = sub_df2.iloc[0]
            temp2 = sub_df2["T"] + 273.15
            print(temp, temp2)
            if abs(temp - temp2) < 15:
                types.append("congruent")
            else:
                types.append("non-congruent")
            valid_idx.append(i)
            congruent_temps.append(temp)
            meta_data[entry_key] = {
                "system": '-'.join(sorted_sys),
                "index": int(i),
                "Fit Type": pred_tag,
                "type": types[-1],
                "mpds_temp": congruent_temp,
                "mpds_phase": congruent_phase,
                "calculated_temp": temp,
                "mae": mae,
                "rmse": rmse,
                "ternary_meta": tern_meta,
                "binary_L_params": binary_L_dict,
            }

            # binary_plot1 = plotter.bin_fig_list[0]
            # ploff.plot(binary_plot1, filename=dump_dir + f'{"-".join(sorted_sys)}_{interp}1_binary.html', auto_open=True)
            print(f"System {tern_sys} with {congruent_phase} index {i} and {temp} is valid")

        except Exception as e:
            system_label = '-'.join(sorted_sys) if sorted_sys is not None else str(tern_sys)
            print(f"Error in system {system_label} with index {i}: {e}")
            Error_dict[entry_key] = {
                "status": "error",
                "system": system_label,
                "index": int(i),
                "phase": str(congruent_phase),
                "message": str(e),
            }

    print(congruent_temps)
    print(types)
    print(valid_idx)
    print(Error_dict)

    status_counts = Counter(v.get("status", "unknown") for v in Error_dict.values())
    print(f"Error/skip status counts: {dict(status_counts)}")

    # Output integrity checks: successful rows and attached result vectors must align.
    assert len(valid_idx) == len(congruent_temps) == len(types), (
        "Output vector length mismatch: "
        f"valid_idx={len(valid_idx)}, congruent_temps={len(congruent_temps)}, types={len(types)}"
    )

    # Keep row-level selection by explicit indices so duplicate chemical systems
    # with different intermetallics are preserved in the final output.
    new_df = ternary_df.iloc[valid_idx].copy()
    new_df["source_row_idx"] = valid_idx
    new_df["system_key"] = new_df["elements"].apply(
        lambda x: '-'.join(sorted(ast.literal_eval(x) if isinstance(x, str) else x))
    )
    new_df["gliq_melting_temp"] = congruent_temps
    new_df["type"] = types

    # Diagnostic confirmation: multiple intermetallic rows per same system should remain.
    duplicated_system_rows = int(new_df["system_key"].duplicated(keep=False).sum())
    print(f"Rows in output belonging to duplicated chemical systems: {duplicated_system_rows}")
    print(new_df)

    new_df.to_excel(os.path.join(dump_dir, f"ternary_Gliq_mps_final_{interp}.xlsx"), index=False)
            
    with open(os.path.join(dump_dir, f"ternary_Gliq_meta_final_{interp}.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    with open(os.path.join(dump_dir, f"ternary_Gliq_errors_final_{interp}.json"), "w") as f:
        json.dump(Error_dict, f, indent=4)

if __name__ == "__main__":
    main()