import os 
from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.config import data_dir 
import gliquid.load_binary_data as lbd
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt
import json


def main():
    dump_dir = "all_dumps/binary_fits"
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    # param_format = 'linear'   
    param_format = 'whs'   

    binary = ['Zr', 'Ga']
    
    binary_sys_name = ("-").join(sorted(binary))

    print(binary_sys_name)

    binary_json, binary_component_data, binary_digitized_liquidus = lbd.load_mpds_data(binary_sys_name, verbose=True)

    print("Component data: ", binary_component_data)
    print("Digitized liquidus; ", binary_digitized_liquidus)

    identified_phases = lbd.identify_mpds_phases(binary_json)
    print(identified_phases)

    binary_dft_ch, _ = lbd.get_dft_convexhull(binary_sys_name, verbose=True)

    binary_pdp = PDPlotter(binary_dft_ch)
    fig = binary_pdp.get_plot()
    fig.update_layout(plot_bgcolor = "white", paper_bgcolor = "white")
    fig.update_layout(width=600, height=500)
    fig.show()

    binary_system = BinaryLiquid.from_cache(binary_sys_name, param_format=param_format)
    binary_plotter = BLPlotter(binary_system)
    binary_plotter.show('pc')
    plt.show()

    fit_results = binary_system.fit_parameters(verbose=True, n_opts=3)[0]
    keys_to_extract = ['mae', 'rmse', 'norm_mae', 'norm_rmse', 'L0_a', 'L0_b', 'L1_a', 'L1_b', 'L0', 'L1']
    dump_dict = {k: fit_results[k] for k in keys_to_extract if k in fit_results}
    print(dump_dict)

    binary_plotter.show('fit+liq')
    binary_plotter.show('nmp')
    plt.show()

    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"{binary_sys_name}_{param_format}_fit_results.json")
    with open(dump_path, "w") as f:
        json.dump(dump_dict, f, indent=4)

if __name__ == "__main__":
    main()