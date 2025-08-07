from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd
import numpy as np
import ast

dump_dir = "all_dumps/gliq_manu/"
read_dir = "all_dumps/binary_fits/"

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def main():
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = "whs"
    interp = "linear"

    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")
    ternary_df = pd.read_excel("data/ternary_dft_data/ternary_im_filtered.xlsx")
    ternary_sys_list = ternary_df["elements"].tolist()
    ternary_sys_list = [ast.literal_eval(e) if isinstance(e, str) else e for e in ternary_sys_list]
    
    system_list = binary_param_df["system"].tolist()
    print(system_list)
    print(ternary_sys_list)
    
    congruent_temps = []
    types = []
    valid_idx = []

    for tern_sys in ternary_sys_list:
        i = ternary_sys_list.index(tern_sys)
        print(f"System {tern_sys} with index {i}")
        congruent_temp = ternary_df.iloc[i]["melting_point_k"]
        congruent_phase = ternary_df.iloc[i]["reduced_formula"]
        try:
            sorted_sys = sorted(tern_sys)
            binary_sys_labels = [
                f"{sorted_sys[0]}-{sorted_sys[1]}",
                f"{sorted_sys[1]}-{sorted_sys[2]}",
                f"{sorted_sys[2]}-{sorted_sys[0]}"
            ]
            print(binary_sys_labels)

            binary_L_dict = {}

            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split("-")))
                print(flipped_sys)

                if bin_sys in binary_param_df["system"].tolist():
                    params = binary_param_df[binary_param_df["system"] == bin_sys].iloc[0]
                elif flipped_sys in binary_param_df["system"].tolist():
                    params = binary_param_df[binary_param_df["system"] == flipped_sys].iloc[0]
                else:
                    raise Exception("System not in df")

                binary_L_dict[bin_sys] = [
                    float(params["L0_a"]),
                    float(params["L0_b"]),
                    float(params["L1_a"]),
                    float(params["L1_b"])
                ]

            print(binary_L_dict)
            plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type=interp, param_format=tern_param_format,
                                        L_dict=binary_L_dict, temp_slider=[0, -300], T_incr=5.0, delta=0.01)
            plotter.interpolate()
            plotter.process_data()
            df_list = plotter.equil_df_list
            concat_df = pd.concat(df_list, ignore_index=True)
            sub_df = concat_df[concat_df["Phase"] == congruent_phase]
            if sub_df.empty:
                raise Exception("MPDS congruent phase not on the hull!")

            sub_df = sub_df.sort_values(by="T", ascending=False)
            sub_df = sub_df.iloc[0]
            comp = [sub_df["x0"], sub_df["x1"]]
            temp = sub_df["T"] + 273.15
            congruent_temps.append(temp)
            print(concat_df)
            sub_df2 = concat_df[(concat_df["Phase"] == "L") &
                                (np.isclose(concat_df["x0"], comp[0], rtol=0, atol=0.025)) &
                                (np.isclose(concat_df["x1"], comp[1], rtol=0, atol=0.025))]
            sub_df2 = sub_df2.sort_values(by="T", ascending=True)
            sub_df2 = sub_df2.iloc[0]
            temp2 = sub_df2["T"] + 273.15
            print(temp, temp2)
            if abs(temp - temp2) < 10:
                types.append("congruent")
            else:
                types.append("non-congruent")
            tern_fig = plotter.plot_ternary()
            ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_system.html', auto_open=False)
            valid_idx.append(i)
            print(f"System {tern_sys} with {congruent_phase} index {i} and {temp} is valid")

        except Exception as e:
            print(f"Error in system {tern_sys} with index {i}: {e}")
            continue

        print(congruent_temps)
        print(types)
        print(valid_idx)

        new_df = ternary_df.iloc[valid_idx]
        new_df["gliq_melting_temp"] = congruent_temps
        new_df["type"] = types
        print(new_df)

        new_df.to_excel(os.path.join(dump_dir, "ternary_Gliq_mps.xlsx"), index=False)
            


if __name__ == "__main__":
    main()