import ast
import itertools
import pandas as pd 


root_dir = "data/ternary_dft_data"
mpds_pull_df = pd.read_excel(f"{root_dir}/ternary_im_melting_points.xlsx")
mpds_IQR_df = pd.read_excel(f"{root_dir}/ternary_im_filtered_IQR.xlsx")
mpds_metals_df = pd.read_excel(f"{root_dir}/ternary_im_filtered.xlsx")
binary_params_df = pd.read_excel(f"{root_dir}/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")

print(len(mpds_pull_df), "MPDS ternary systems pulled from MPDS")
print(len(mpds_IQR_df), "MPDS ternary systems after IQR filtering")
print(len(mpds_metals_df), "MPDS ternary systems after metals filtering")
print(len(binary_params_df), "binary systems with fitted parameters")


only_include = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Eu', 
                'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 
                'Nd', 'Ni', 'Os', 'Pb', 'Pr', 'Rb', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Th', 
                'Ti', 'Tl', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

mpds_IQR_filtered_df = mpds_IQR_df[
    mpds_IQR_df["elements"].apply(lambda elements: all(element in only_include for element in ast.literal_eval(elements)))
]

print("number of mpds congruent melting points data:", len(mpds_IQR_filtered_df))

binary_systems = set(binary_params_df["system"].apply(lambda system: tuple(sorted(system.split("-")))))
binary_elements = sorted(set(element for system in binary_systems for element in system))
hypothetical_ternaries = [
    ternary for ternary in itertools.combinations(binary_elements, 3)
    if all(tuple(sorted(pair)) in binary_systems for pair in itertools.combinations(ternary, 2))
]
print("number of hypothetical ternaries with all binary fits:", len(hypothetical_ternaries))

mpds_IQR_fit_df = mpds_IQR_filtered_df[
    mpds_IQR_filtered_df["elements"].apply(
        lambda elements: all(tuple(sorted(pair)) in binary_systems for pair in itertools.combinations(ast.literal_eval(elements), 2))
    )
]
print("number of mpds congruent melting points with all binary fits:", len(mpds_IQR_fit_df))

