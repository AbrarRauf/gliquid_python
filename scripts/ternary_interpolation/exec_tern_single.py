from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json

dump_dir = "all_dumps/ternary_htmls/"
read_dir = "all_dumps/binary_fits/"

def plot_ternary_system():
    # Bi-Cd-Sn system
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_sys = ["Bi", "Zr", "Te"]
    bin_param_format = 'whs'
    tern_param_format = 'whs'
    # bin_param_format = 'linear'
    # tern_param_format = 'linear'

    sorted_sys = sorted(tern_sys)
    binary_sys_labels = [
        f"{sorted_sys[0]}-{sorted_sys[1]}",
        f"{sorted_sys[1]}-{sorted_sys[2]}",
        f"{sorted_sys[2]}-{sorted_sys[0]}"
    ]

    print(binary_sys_labels)

    binary_L_dict = {}

    for sys in binary_sys_labels:
        flipped_sys = "-".join(sorted(sys.split("-")))
        print(flipped_sys)
        json_path = os.path.join(read_dir, f"{sys}_{bin_param_format}_fit_results.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(read_dir, f"{flipped_sys}_{bin_param_format}_fit_results.json")
            
        with open(json_path, "r") as f:
            params = json.load(f)
        binary_L_dict[sys] = [
            params["L0_a"],
            params["L0_b"],
            params["L1_a"],
            params["L1_b"]
        ]

    print(binary_L_dict)
    
    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=tern_param_format,
                                  L_dict=binary_L_dict, temp_slider=[0, -300], T_incr=5, delta=0.025)
    plotter.interpolate()
    plotter.process_data()
    tern_fig = plotter.plot_ternary()
    # ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_{tern_param_format}_system.html', auto_open=True)
    ploff.plot(tern_fig, filename=dump_dir + f'{"-".join(sorted_sys)}_tielines_system.html', auto_open=True)


if __name__ == "__main__":
    plot_ternary_system()
