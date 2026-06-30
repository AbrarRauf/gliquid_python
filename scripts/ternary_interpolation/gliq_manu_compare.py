import pandas as pd 
import ast

df_new = pd.read_excel("all_dumps/gliq_manu_forreal/ternary_Gliq_mps_final_linear.xlsx")
df_old = pd.read_excel("all_dumps/gliq_manu_test7_linear/ternary_Gliq_mps_final_linear_filtered.xlsx")
param_file = "all_dumps/param_set/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx"
df_params = pd.read_excel(param_file)

required_cols = {"reduced_formula", "elements", "melting_point_k", "gliq_melting_temp"}
missing_cols = {
    "df_new": required_cols - set(df_new.columns),
    "df_old": required_cols - set(df_old.columns),
}
missing_cols = {name: cols for name, cols in missing_cols.items() if cols}
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
if "system" not in df_params.columns:
    raise ValueError(f"Missing required param column: system")


def formula_set(df):
    return set(df["reduced_formula"].dropna().astype(str))


def by_formula(df):
    clean = df[["reduced_formula", "elements", "melting_point_k", "gliq_melting_temp"]].copy()
    clean["reduced_formula"] = clean["reduced_formula"].astype(str)
    clean["melting_point_k"] = pd.to_numeric(clean["melting_point_k"], errors="coerce")
    clean["gliq_melting_temp"] = pd.to_numeric(clean["gliq_melting_temp"], errors="coerce")
    return clean.groupby("reduced_formula", as_index=False).agg(
        elements=("elements", "first"),
        melting_point_k=("melting_point_k", "mean"),
        gliq_melting_temp=("gliq_melting_temp", "mean"),
    )


def elements_tuple(elements):
    if isinstance(elements, str):
        elements = ast.literal_eval(elements)
    return tuple(sorted(elements))


def ternary_system(elements):
    return "-".join(elements_tuple(elements))


def required_binary_systems(elements):
    elements = elements_tuple(elements)
    return [f"{elements[0]}-{elements[1]}", f"{elements[0]}-{elements[2]}", f"{elements[1]}-{elements[2]}"]


def normalize_binary_system(system):
    elements = str(system).split("-")
    return "-".join(sorted(elements)) if len(elements) == 2 else str(system)


new_formulas = formula_set(df_new)
old_formulas = formula_set(df_old)
common_formulas = sorted(new_formulas & old_formulas)
old_only_formulas = sorted(old_formulas - new_formulas)
new_only_formulas = sorted(new_formulas - old_formulas)

print("\nReduced formula coverage")
print(f"df_new unique formulas: {len(new_formulas)}")
print(f"df_old unique formulas: {len(old_formulas)}")
print(f"common formulas: {len(common_formulas)}")
print(f"df_new subset of df_old: {new_formulas <= old_formulas}")
print(f"\nin df_old but missing from df_new ({len(old_only_formulas)}):")
print(old_only_formulas)
print(f"\nin df_new but missing from df_old ({len(new_only_formulas)}):")
print(new_only_formulas)

compare = by_formula(df_new).merge(
    by_formula(df_old),
    on="reduced_formula",
    how="inner",
    suffixes=("_new", "_old"),
)
compare["gliq_delta_k"] = compare["gliq_melting_temp_new"] - compare["gliq_melting_temp_old"]
compare["abs_gliq_delta_k"] = compare["gliq_delta_k"].abs()
compare["melting_point_delta_k"] = compare["melting_point_k_new"] - compare["melting_point_k_old"]
compare["new_vs_melting_point_k"] = compare["gliq_melting_temp_new"] - compare["melting_point_k_new"]
compare["old_vs_melting_point_k"] = compare["gliq_melting_temp_old"] - compare["melting_point_k_old"]
compare = compare.sort_values("abs_gliq_delta_k", ascending=False)
stats = compare["gliq_delta_k"].agg(["count", "mean", "std", "min", "median", "max"])
stats["mean_abs"] = compare["abs_gliq_delta_k"].mean()
mp_stats = compare[["new_vs_melting_point_k", "old_vs_melting_point_k"]].agg(
    ["count", "mean", "std", "min", "median", "max"]
)
mp_stats.loc["mean_abs"] = compare[["new_vs_melting_point_k", "old_vs_melting_point_k"]].abs().mean()

print("\nCommon-formula gliq_melting_temp deviations (new - old, K)")
compare_print = compare[[
    "reduced_formula",
    "melting_point_k_new",
    "melting_point_k_old",
    "melting_point_delta_k",
    "gliq_melting_temp_new",
    "gliq_melting_temp_old",
    "gliq_delta_k",
    "new_vs_melting_point_k",
    "old_vs_melting_point_k",
]].copy()
compare_print.loc["AVERAGE"] = ""
compare_print.loc["AVERAGE", "reduced_formula"] = "AVERAGE"
compare_print.loc["AVERAGE", [
    "melting_point_delta_k",
    "gliq_delta_k",
    "new_vs_melting_point_k",
    "old_vs_melting_point_k",
]] = compare[[
    "melting_point_delta_k",
    "gliq_delta_k",
    "new_vs_melting_point_k",
    "old_vs_melting_point_k",
]].abs().mean()
print(compare_print.to_string(index=False))

# extract compare_print df to excel
compare_print.to_excel("all_dumps/gliq_manu_forreal/compare_print.xlsx", index=False)

print("\nDeviation statistics (K)")
print(stats.to_string())
print("\nDeviation from melting_point_k statistics (K)")
print(mp_stats.to_string())

old_systems = set(df_old["elements"].dropna().apply(ternary_system))
new_systems = set(df_new["elements"].dropna().apply(ternary_system))
dropped_systems = sorted(old_systems - new_systems)
new_only_systems = sorted(new_systems - old_systems)

print("\nTernary system coverage from elements")
print(f"df_new unique ternary systems: {len(new_systems)}")
print(f"df_old unique ternary systems: {len(old_systems)}")
print(f"ternary systems dropped from df_new ({len(dropped_systems)}):")
print(dropped_systems)
print(f"ternary systems only in df_new ({len(new_only_systems)}):")
print(new_only_systems)

removed_elements = {"K", "As", "Se", "Sb", "Te", "Cs", "Pd", "Pt", "U", "Pu"}
param_systems = set(df_params["system"].dropna().apply(normalize_binary_system))
dropped_formula_diag = by_formula(df_old[df_old["reduced_formula"].astype(str).isin(old_only_formulas)])
dropped_formula_diag["ternary_system"] = dropped_formula_diag["elements"].apply(ternary_system)
dropped_formula_diag["removed_elements"] = dropped_formula_diag["elements"].apply(
    lambda x: sorted(set(elements_tuple(x)) & removed_elements)
)
dropped_formula_diag["dropped_by_removed_element"] = dropped_formula_diag["removed_elements"].str.len().gt(0)
dropped_formula_diag["required_binary_systems"] = dropped_formula_diag["elements"].apply(required_binary_systems)
dropped_formula_diag["missing_binary_systems"] = dropped_formula_diag["required_binary_systems"].apply(
    lambda systems: [system for system in systems if system not in param_systems]
)
dropped_formula_diag["dropped_by_missing_binary_fit"] = dropped_formula_diag["missing_binary_systems"].str.len().gt(0)
dropped_formula_diag["drop_reason"] = "unexplained"
dropped_formula_diag.loc[dropped_formula_diag["dropped_by_missing_binary_fit"], "drop_reason"] = "missing binary fit"
dropped_formula_diag.loc[dropped_formula_diag["dropped_by_removed_element"], "drop_reason"] = "removed element"
not_removed = dropped_formula_diag[~dropped_formula_diag["dropped_by_removed_element"]]
unexplained = not_removed[~not_removed["dropped_by_missing_binary_fit"]]

print("\nOld-only formula drop diagnosis")
print(f"removed elements: {sorted(removed_elements)}")
print(f"df_new param file: {param_file}")
print(f"old-only formulas with removed elements: {dropped_formula_diag['dropped_by_removed_element'].sum()}")
print(f"old-only formulas without removed elements but missing binary fits: {not_removed['dropped_by_missing_binary_fit'].sum()}")
print(f"old-only formulas not explained by either check: {len(unexplained)}")
print(dropped_formula_diag[[
    "reduced_formula",
    "ternary_system",
    "removed_elements",
    "required_binary_systems",
    "missing_binary_systems",
    "drop_reason",
]].to_string(index=False))

print("\nHypothesis check")
if unexplained.empty:
    print("Every old-only formula not dropped by removed elements is missing at least one binary fit in the df_new param file.")
else:
    print("These old-only formulas were not dropped by removed elements and have all required binary fits:")
    print(unexplained[["reduced_formula", "ternary_system", "required_binary_systems"]].to_string(index=False))
