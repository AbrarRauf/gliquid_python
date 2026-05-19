import pandas as pd
import itertools
from ast import literal_eval

binary_param_df = pd.read_excel("data/ternary_dft_data/tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx")

# -----------------------------------------------------------------------------
# Category filter controls (edit these to explore different invariant mixes)
# -----------------------------------------------------------------------------
# Any listed column can be used as a category in KEEP_CATEGORY_COMBOS.

# CATEGORY_COLUMNS = [
#     "euts", "cmps", "pers", "migs",
# ]

CATEGORY_COLUMNS = [
    "mpds_euts", "mpds_cmps", "mpds_pers", "mpds_migs",
]

# Keep rows whose NON-EMPTY category set matches one of these combos.
# Examples:
# 1) Only MPDS eutectics:
#    KEEP_CATEGORY_COMBOS = [["mpds_euts"]]
# 2) Only fitted eutectics:
#    KEEP_CATEGORY_COMBOS = [["euts"]]
# 3) Fitted eutectics + peritectics only:
#    KEEP_CATEGORY_COMBOS = [["euts", "pers"]]
# 4) Allow multiple exact combinations:
#    KEEP_CATEGORY_COMBOS = [["euts"], ["euts", "pers"], ["mpds_euts"]]

KEEP_CATEGORY_COMBOS = [["mpds_euts"]]
# Match behavior:
# - "exact": row categories must equal one keep combo exactly.
# - "contains": row categories must contain all categories in at least one combo.
CATEGORY_MATCH_MODE = "exact"

only_include = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Eu', 
                'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 
                'Nd', 'Ni', 'Os', 'Pb', 'Pr', 'Rb', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Th', 
                'Ti', 'Tl', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']


def parse_list_like(value):
    """Convert text/object cell values to Python lists where possible."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        try:
            parsed = literal_eval(text)
            if isinstance(parsed, list):
                return parsed
            # Wrap non-list parsed objects to preserve non-empty semantics.
            return [parsed]
        except (ValueError, SyntaxError):
            return []
    return []


def parse_binary_system(system_value):
    """Return a sorted 2-element tuple for valid binary labels like A-B."""
    if not isinstance(system_value, str):
        return None
    parts = [p.strip() for p in system_value.split("-") if p.strip()]
    if len(parts) != 2:
        return None
    return tuple(sorted(parts))


def row_nonempty_categories(row, category_cols):
    """Return the set of category column names that are non-empty in a row."""
    return {col for col in category_cols if len(row[col]) > 0}


def keep_row_by_category_combo(row_categories, keep_combos, mode="exact"):
    """Evaluate whether a row category set should be kept."""
    if mode == "exact":
        return row_categories in keep_combos
    if mode == "contains":
        return any(combo.issubset(row_categories) for combo in keep_combos)
    raise ValueError(f"Unsupported CATEGORY_MATCH_MODE: {mode}")


def main():
    df = binary_param_df.copy()

    # Validate and parse category columns used in combo filtering.
    keep_combo_sets = [frozenset(combo) for combo in KEEP_CATEGORY_COMBOS]
    referenced_cols = set().union(*keep_combo_sets) if keep_combo_sets else set()
    invalid_references = sorted(col for col in referenced_cols if col not in CATEGORY_COLUMNS)
    if invalid_references:
        raise ValueError(f"Columns in KEEP_CATEGORY_COMBOS not in CATEGORY_COLUMNS: {invalid_references}")

    for col in CATEGORY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].apply(parse_list_like)

    if "system" not in df.columns:
        raise ValueError("Missing required column: system")

    df["binary_tuple"] = df["system"].apply(parse_binary_system)
    df = df[df["binary_tuple"].notna()].copy()

    include_set = set(only_include)
    df = df[
        df["binary_tuple"].apply(lambda pair: pair[0] in include_set and pair[1] in include_set)
    ].copy()

    # Keep rows that match at least one user-declared category combination.
    df["nonempty_categories"] = df.apply(
        lambda row: row_nonempty_categories(row, CATEGORY_COLUMNS), axis=1
    )
    df = df[
        df["nonempty_categories"].apply(
            lambda cat_set: keep_row_by_category_combo(cat_set, keep_combo_sets, CATEGORY_MATCH_MODE)
        )
    ].copy()

    unique_binary_pairs = set(df["binary_tuple"].tolist())
    fitted_binary_count = len(unique_binary_pairs)

    # Build candidate ternaries from all elements that appear in valid binary eutectic systems.
    element_pool = sorted({elem for pair in unique_binary_pairs for elem in pair})
    ternary_systems = []
    for a, b, c in itertools.combinations(element_pool, 3):
        pair_ab = tuple(sorted((a, b)))
        pair_bc = tuple(sorted((b, c)))
        pair_ac = tuple(sorted((a, c)))
        if pair_ab in unique_binary_pairs and pair_bc in unique_binary_pairs and pair_ac in unique_binary_pairs:
            ternary_systems.append("-".join((a, b, c)))

    print(f"Category combinations kept: {[sorted(list(s)) for s in keep_combo_sets]}")
    print(f"Category match mode: {CATEGORY_MATCH_MODE}")
    print(f"Total fitted binary systems after category filtering: {fitted_binary_count}")
    print(f"Total ternary systems that can be interpolated: {len(ternary_systems)}")
    print("Interpolable ternary systems:")
    for system in ternary_systems:
        print(system)


if __name__ == "__main__":
    main()




