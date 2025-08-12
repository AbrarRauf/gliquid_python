import json
import pandas as pd
import itertools
import os
from gliquid.binary import BinaryLiquid, BLPlotter
import gliquid.config as config
from pathlib import Path

os.environ["NEW_MP_API_KEY"] = "Jcw46im7UV1xOfHzbZZ8nkq8BH00Pf6s"
os.environ["MPDS_API_KEY"] = "29r3PRLsIW8YNzxxTCVgDmxP0ylZqfWSUvBVE3IVyrZvpqKG"
# Override python package config settings to use cached data in 'matrix_data' directory
config.set_project_root(Path.cwd().parent) # Dropbox G_liquid directory
config.set_data_dir(Path(config.project_root / "matrix_data")) # Data isin the matrix_data directory
config.set_dir_structure(config._DIR_STRUCT_OPTS[1]) # Nested directory structure

fit_data_dir = f"{config.project_root}/binary_fitting_data"
matrix_dir = config.data_dir
pformat = 'combined_no_1S'  # Parameter format for BinaryLiquid

xlsxfile = f"{fit_data_dir}/multi_fits_combined_no_1S_model-tau8k.xlsx"

elements = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Eu', 'Yb', 'Lu', 'Tm', 'Y', 'Er', 'Ho',
            'Dy', 'Tb', 'Gd', 'Sm', 'Nd', 'Pr', 'Ce', 'La', 'Th', 'U', 'Pu', 'Sc', 'Zr', 'Hf',
            'Ti', 'Ta', 'Nb', 'V', 'Cr', 'Mo', 'W', 'Re', 'Os', 'Ru', 'Ir', 'Rh', 'Pt', 'Pd', 'Au',
            'Ag', 'Cu', 'Ni', 'Co', 'Fe', 'Mn', 'Mg', 'Zn', 'Cd', 'Hg', 'Be', 'Al', 'Ga', 'In', 'Tl',
            'Pb', 'Sn', 'Ge', 'Si', 'B', 'C', 'As', 'Sb', 'Bi', 'Te', 'Se']

system_combos = [sorted([a, b]) for a, b in itertools.combinations(elements, 2)]

default_fields = {
    'system': ["none"],
    'dft_functional': ["none"],
    'mpds_reference': ["none"],
    # 'pd_index': ["none"], # Assume always 0 for now
    'jcode': ["none"],
    'errors': ["none"],
    'ignored_comp_ranges': ["none"],
    'sum_ignrd_comp_range': [0.0],
    'fit_algo': ["none"],
    'fit_constrs': ["none"],
    'fit_n_iters': [0],
    'L0_a': [0.0],
    'L0_b': [0.0],
    'L1_a': [0.0],
    'L1_b': [0.0],
    'L0': [0.0],
    'L1': [0.0],
    'mae': [0.0],
    'norm_mae': [0.0],
    'rmse': [0.0],
    'norm_rmse': [0.0],
    'euts': [json.dumps([])],
    'cmps': [json.dumps([])],
    'pers': [json.dumps([])],
    'migs': [json.dumps([])],
    'mpds_euts': [json.dumps([])],
    'mpds_cmps': [json.dumps([])],
    'mpds_pers': [json.dumps([])],
    'mpds_migs': [json.dumps([])],
    'nmpath': [json.dumps([])],
}


def init_row_df(**kwargs):
    rdf = pd.DataFrame(default_fields)
    for colname, val in kwargs.items():
        if colname in default_fields.keys() and val is not None:
            rdf.at[0, colname] = val
    return rdf


try:
    print("Loading existing data from", xlsxfile)
    df = pd.read_excel(xlsxfile)
    last_fitted_sys = df['system'].drop_duplicates().values[-1]
    print("Last system in fitting progess:", last_fitted_sys)

    # Remove ALL rows with the last system and begin fitting from that system to ensure that all fits were finished
    df = df[df['system'] != last_fitted_sys]
    sys_combo = last_fitted_sys.split('-')
    system_combos = system_combos[system_combos.index(sys_combo):]
except FileNotFoundError:
    print("No existing data found. Creating new dataframe.")
    df = init_row_df()
    df = df.drop(index=0)

for i, components in enumerate(system_combos):
    sys_name = '-'.join(sorted(components))
    pd_ind = 0

    print(f"\n\033[4m{sys_name}\033[0m")
    # print(f"pd index = {pd_ind}")

    bl = BinaryLiquid.from_cache(sys_name, param_format=pformat, pd_ind=pd_ind, comp_range_fit_lim=0.8)

    if bl.mpds_json['reference'] is None:
        mpds_ref = "none"
        mpds_jcode = "none"
    else:
        mpds_ref = bl.mpds_json['reference']['entry']
        if 'jcode' in bl.mpds_json:
            mpds_jcode = bl.mpds_json['jcode']
        else:
            mpds_jcode = "not recorded"

    if bl.init_error:
        row_df = init_row_df(system=sys_name, pd_index=pd_ind, mpds_reference=mpds_ref, jcode=mpds_jcode,
                            dft_functional=bl.dft_type, errors='missing data init error')
        df = pd.concat([df, row_df])
        continue

    fit_data = bl.fit_parameters(n_opts=7, verbose=True, check_phase_mismatch=False)
   
    if not fit_data:
        err = 'missing phases init error' if bl.init_error else 'fitting conditions error' 
        row_df = init_row_df(system=sys_name, pd_index=pd_ind, mpds_reference=mpds_ref, jcode=mpds_jcode,
                            dft_functional=bl.dft_type,
                             errors=err)
        df = pd.concat([df, row_df])
        continue

    if bl.ignored_comp_ranges:
        ignored_comp_ranges = ", ".join(["-".join([str(round(i, 3)) for i in crge])
                                         for crge in bl.ignored_comp_ranges])
        sum_ignrd_comp_range = sum([abs(crge[1] - crge[0]) for crge in bl.ignored_comp_ranges])
    else:
        ignored_comp_ranges = "none"
        sum_ignrd_comp_range = 0.0

    exp_invs = bl.invariants
    mpds_euts = json.dumps([inv for inv in exp_invs if inv['type'] == 'eut'])
    mpds_pers=json.dumps([inv for inv in exp_invs if inv['type'] == 'per'])
    mpds_cmps=json.dumps([inv for inv in exp_invs if inv['type'] == 'cmp'])
    mpds_migs=json.dumps([inv for inv in exp_invs if inv['type'] == 'mig'])

    for fit in fit_data:  
        row_df = init_row_df(system=sys_name, pd_index=pd_ind, mpds_reference=mpds_ref, jcode=mpds_jcode,
                             ignored_comp_ranges=ignored_comp_ranges, sum_ignrd_comp_range=sum_ignrd_comp_range,
                             fit_algo=fit['algo'], fit_constrs=json.dumps(fit['constrs']), fit_n_iters=fit['n_iters'],
                             euts=json.dumps(fit['euts']), pers=json.dumps(fit['pers']),
                             cmps=json.dumps(fit['cmps']), migs=json.dumps(fit['migs']),
                             mpds_euts=mpds_euts, mpds_pers=mpds_pers, mpds_cmps=mpds_cmps, mpds_migs=mpds_migs,
                             mae=fit['mae'], norm_mae=fit['norm_mae'], rmse=fit['rmse'], norm_rmse=fit['norm_rmse'],
                             L0_a=fit['L0_a'], L0_b=fit['L0_b'], L1_a=fit['L1_a'], L1_b=fit['L1_b'],
                             L0=fit['L0'], L1=fit['L1'], dft_functional=bl.dft_type, nmpath=json.dumps(fit['nmpath'].tolist()))
        df = pd.concat([df, row_df])
   
    # blp = BLPlotter(bl)
    # bl.hsx.plot_hsx().show()
    # blp.write_image(plot_type='fit+liq', stream=f"{matrix_dir}/{sys_name}/{sys_name}_FITTED_MP_GGA-NO1S.svg")
    # blp.write_image(plot_type='nmp', stream=f"{matrix_dir}/{sys_name}/{sys_name}_NMPATH_MP_GGA-NO1S.svg",
    #                 plot_a_params=True)
    # blp.write_image(plot_type='ch+g', stream=f"{matrix_dir}/{sys_name}/{sys_name}_ENTRIES_MP_GGA+GFIT-NO1S.svg")
    # blp.show(plot_type='fit+liq')
    # blp.show(plot_type='nmp')
    # blp.show(plot_type='ch+g')
    df.to_excel(xlsxfile, index=False)

print(f"\n\033[1mFitting complete. Data saved to {xlsxfile}\033[0m")
df.to_excel(xlsxfile, index=False)
   