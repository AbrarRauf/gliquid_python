import os 
from gliquid.binary import BinaryLiquid, BLPlotter
from gliquid.config import data_dir 
import gliquid.load_binary_data as lbd
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
param_format = 'linear'   

binary_sys_name = 'Te-Zr'

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

for field, value in fit_results.items():
    if isinstance(value, float):
        print(f'{field.upper()}: {value:.3f}')

binary_plotter.show('fit+liq')
binary_plotter.show('nmp')
plt.show()