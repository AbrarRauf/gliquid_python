import os
from matplotlib import pyplot as plt
import plotly.offline as ploff
from gliquid.config import data_dir
from gliquid.binary import BinaryLiquid, BLPlotter
from ternary_interpolation.ternary_HSX import ternary_gtx_plotter


def fit_c_fe_system():
    c_fe_system = BinaryLiquid.from_cache("C-Fe", param_format='whs')
    print(c_fe_system.mpds_json['reference']['entry'])
    c_fe_system.phases.insert(1, {'name': '(Fe) ht', 'comp': 0.92, 'energy': -0.01, 'points': []}) # Add max sol ht Fe phase
    print(c_fe_system.phases)


    # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
    fit_res = c_fe_system.fit_parameters(verbose=True, n_opts=3)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    blp = BLPlotter(c_fe_system)
    blp.show('fit+liq')
    # blp.get_plot('nmp', plot_a_params=True)
    # plt.show()

def fit_c_ni_system():
    c_ni_system = BinaryLiquid.from_cache("C-Ni", param_format='whs')
    print(c_ni_system.mpds_json['reference']['entry'])

    # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
    fit_res = c_ni_system.fit_parameters(verbose=True, n_opts=3)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    blp = BLPlotter(c_ni_system)
    # blp.show('fit+liq')
    blp.get_plot('nmp', plot_a_params=True)
    plt.show()


def fit_fe_ga_system():
    fe_ga_system = BinaryLiquid.from_cache("Fe-Ga", param_format='whs')
    print(fe_ga_system.mpds_json['reference']['entry'])

    fe_ga_system.ignored_comp_ranges.append([0, 0.48])
    fe_ga_system.phases.insert(2, {'name': 'Ga4Fe3', 'comp': 0.571, 'energy': -29000, 'points': []})
    fe_ga_system.phases.insert(2, {'name': 'GaFe3 Ga+ ht', 'comp': 0.35, 'energy': -30000, 'points': []}) 
    fe_ga_system.phases[1]['energy'] = -19000
    print(fe_ga_system.phases)

    fit_res = fe_ga_system.fit_parameters(verbose=True, n_opts=3)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    blp = BLPlotter(fe_ga_system)
    blp.show('fit+liq')
    blp.get_plot('nmp', plot_a_params=True)
    plt.show()


def correct_fe_ga_system_phases():
    # fitted_params = [-72905, -10.46, -44951, 0]  # These are the fitted parameters from the previous fit
    fitted_params = [-81580, -2.00, -44018, 0]
    fe_ga_system = BinaryLiquid.from_cache("Fe-Ga", param_format='whs', params=fitted_params)
    print(fe_ga_system.phases)
    # Add max sol ht Fe phase
    fe_ga_system.phases.pop(1)  # Remove GaFe3
    fe_ga_system.phases.insert(1, {'name': 'GaFe3 Ga+ ht', 'comp': 0.35, 'energy': -30000, 'points': []}) 
    fe_ga_system.phases.insert(2, {'name': 'Ga4Fe3', 'comp': 0.571, 'energy': -29000, 'points': []})
    print(fe_ga_system.phases)
    fe_ga_system.update_phase_points()

    blp = BLPlotter(fe_ga_system)
    blp.show('fit+liq')

def predict_c_ga_system(method=1):
    if method == 1: # Use predicted parameters for 4-param linear model
        pform = 'linear'
        c_ga_params = [-64640, 7.3, 57820, -37.98] # -7.3 or 7.3? Could try 0
    elif method == 2: # Use Al-C parameters and transmute
        pform = 'whs' # can be whatever works best
        print("transmuting Al-C parameters to C-Ga")
        
        al_c_system = BinaryLiquid.from_cache("Al-C", param_format=pform)
        print(al_c_system.mpds_json['reference']['entry'])

        # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
        fit_res = al_c_system.fit_parameters(verbose=True, n_opts=1)
        for fit in fit_res:
            fit.pop('nmpath', None)
            print(fit)

        blp = BLPlotter(al_c_system)
        blp.show('fit+liq')
        blp.get_plot('nmp', plot_a_params=True)
        plt.show()
        c_ga_params = al_c_system.get_params()
        c_ga_params[2] *= -1

    c_ga_system = BinaryLiquid.from_cache("C-Ga", param_format=pform, params=c_ga_params)
    BLPlotter(c_ga_system).show('pred')


def plot_CGaFe_system():
    tern_sys = ["C", "Fe", "Ga"]
    binary_L_dict = {"C-Fe": [-160695, 35.19, -18327, 0],
                     "Fe-Ga": [-84504, -0.407, -45587, 0],
                     "Ga-C": [217850, -113.56, -43585, 0]}
    
    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format="whs",
                                  L_dict=binary_L_dict, temp_slider=[0, -300], T_incr=5)
    # print(plotter)
    plotter.interpolate()
    print(plotter.hsx_df)
    plotter.hsx_df = plotter.hsx_df[plotter.hsx_df['Phase Name'] != 'GaFe3']
    new_gafe3_phase = {'x0': 1-0.35, 'x1': 0.35, 'S': 0, 'H': -30000, 'Phase Name': 'GaFe3 Ga+ ht'}
    plotter.hsx_df = plotter.hsx_df._append(new_gafe3_phase, ignore_index=True)
    ga4fe3_phase = {'x0': 1-0.571, 'x1': 0.571, 'S': 0, 'H': -29000, 'Phase Name': 'Ga4Fe3'}
    plotter.hsx_df = plotter.hsx_df._append(ga4fe3_phase, ignore_index=True)
    plotter.process_data()
    tern_fig = plotter.plot_ternary()
    ploff.plot(tern_fig, filename='./figures/CFeGa_system.html', auto_open=True)


if __name__ == "__main__":
    # Fit parameters for the following binary systems:
    # C-Ni
    # C-Fe
    # C-Ga -> DNE, try Al-C instead and transmute parameters. Or, use predicted parameters.
    os.environ["NEW_MP_API_KEY"] = "YOUR API KEY HERE"
    # fit_c_fe_system()
    # fit_c_ni_system()
    # predict_c_ga_system(method=2)
    # fit_fe_ga_system()
    # correct_fe_ga_system_phases()
    plot_CGaFe_system()