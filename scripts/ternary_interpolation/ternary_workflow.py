"""Script-level ternary workflow using General HSX interpolation + plotting.

This script is intentionally configured via in-file variables (no CLI args).
Update the USER CONFIG section and run the file directly.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd

from general_HSX import GeneralEquilibrium, GeneralInterpolation
from general_plotter import PhaseBoundaryPlotter


# ============================================================================
# USER CONFIG
# ============================================================================

# User-specified ternary system (order does not matter; internally sorted).
TERNARY_SYSTEM: List[str] = ["Ce", "Fe", "Si"]

# Information tags.
INCLUDE_POLYMORPHS: bool = True
INCLUDE_SOLID_SOLUTIONS: bool = False

# Interpolation and equilibrium settings.
GRID_DELTA: float = 0.025
TEMP_DELTA_K: float = 10.0
# Example: (900.0, 2600.0). None -> auto bounds from fusion data.
TEMP_BOUNDS_K = None
# Interpolation scheme options from general_HSX: 'linear', 'muggianu', 'kohler'.
INTERPOLATION_SCHEME: str = "linear"
VERTICAL_SIMPLICES: bool = False
KEEP_SINGLE_PHASE_LIQUID_SIMPLICES: bool = True
USE_MP_CACHE: bool = True

# Default behavior requested: validate all temperature slices.
VALIDATE_ALL_TEMPERATURE_SLICES: bool = True

# Binary parameter sources.
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_PARAM_XLSX = REPO_ROOT / "data" / "ternary_dft_data" / "tau_penalty_s0.005_p8.5_med_sc-filtered-matrix.xlsx"

# Output locations.
OUTPUT_DIR = REPO_ROOT / "all_dumps" / "ternary_workflow"
OUTPUT_PLOT_PATH = OUTPUT_DIR / f"{'_'.join(TERNARY_SYSTEM)}_{INTERPOLATION_SCHEME}_tx_plot.html"


# ============================================================================
# Helpers
# ============================================================================

def _load_binary_params_for_pair(
	pair_label: str,
	primary_df: pd.DataFrame,
) -> List[float]:
	"""Return canonical [L0_a, L0_b, L1_a, L1_b] for one binary pair.

	If only reversed order is present in source data, L1 terms are sign-flipped
	to preserve canonical sorted pair convention.
	"""
	canonical = "-".join(sorted(pair_label.split("-")))
	p0, p1 = canonical.split("-")
	reversed_label = f"{p1}-{p0}"

	def from_df(df: pd.DataFrame):
		if canonical in df["system"].tolist():
			row = df[df["system"] == canonical].iloc[0]
			return [
				float(row["L0_a"]),
				float(row["L0_b"]),
				float(row["L1_a"]),
				float(row["L1_b"]),
			]
		if reversed_label in df["system"].tolist():
			row = df[df["system"] == reversed_label].iloc[0]
			return [
				float(row["L0_a"]),
				float(row["L0_b"]),
				-float(row["L1_a"]),
				-float(row["L1_b"]),
			]
		return None

	params = from_df(primary_df)
	if params is None:
		raise ValueError(f"Missing binary parameters for pair: {pair_label}")
	return params


def _validate_temperature_coverage(eq_solver: GeneralEquilibrium) -> None:
	"""Raise if any temperature slice failed during equilibrium solve."""
	progress = eq_solver.get_temperature_progress()
	total = int(progress.get("total_temperature_slices", 0))
	successful = int(progress.get("n_successful", 0))
	failed = int(progress.get("n_failed", 0))

	if total == 0:
		raise RuntimeError("Equilibrium solve returned zero temperature slices.")
	if failed > 0 or successful != total:
		failed_t = progress.get("failed_temperature_slices", [])
		raise RuntimeError(
			"Equilibrium validation failed: not all temperature slices solved. "
			f"successful={successful}, total={total}, failed={failed}, failed_slices={failed_t[:10]}"
		)


# ============================================================================
# Main workflow
# ============================================================================

def main() -> None:
	elements = sorted(TERNARY_SYSTEM)
	if len(elements) != 3:
		raise ValueError(f"TERNARY_SYSTEM must contain exactly 3 elements, got {elements}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	if not PRIMARY_PARAM_XLSX.exists():
		raise FileNotFoundError(f"Primary parameter file not found: {PRIMARY_PARAM_XLSX}")

	primary_df = pd.read_excel(PRIMARY_PARAM_XLSX)

	binary_pairs = [
		f"{elements[i]}-{elements[j]}"
		for i, j in combinations(range(len(elements)), 2)
	]

	binary_L_dict: Dict[str, List[float]] = {
		pair: _load_binary_params_for_pair(pair, primary_df)
		for pair in binary_pairs
	}

	interp = GeneralInterpolation(
		elements=elements,
		output_dir=str(OUTPUT_DIR),
		grid_delta=GRID_DELTA,
		temp_delta_k=TEMP_DELTA_K,
		temp_bounds_k=TEMP_BOUNDS_K,
		interp_scheme=INTERPOLATION_SCHEME,
		include_polymorphs=INCLUDE_POLYMORPHS,
		include_ref_solid_solutions=INCLUDE_SOLID_SOLUTIONS,
	)
	interp.set_binary_params(binary_L_dict)
	interp.interpolate(use_mp_cache=USE_MP_CACHE)

	eq_solver = GeneralEquilibrium(interp.gtx_data)
	eq_solver.solve(
		vertical_simplices=VERTICAL_SIMPLICES,
		print_progress=True,
		progress_every=1,
		keep_single_phase_liquid_simplices=KEEP_SINGLE_PHASE_LIQUID_SIMPLICES,
	)

	if VALIDATE_ALL_TEMPERATURE_SLICES:
		_validate_temperature_coverage(eq_solver)

	plotter = PhaseBoundaryPlotter(
		equilibrium_df=eq_solver.equilibrium_df,
		element_names=elements,
	)

	# In this API, dependent composition corresponds to elements[0], and
	# x0/x1 correspond to elements[1]/elements[2].
	fig = plotter.plot_ternary_slice(
		comp_a=elements[1],
		comp_b=elements[2],
		comp_c=elements[0],
		fixed_components={},
		phase_extrema_filter=True,
		ternary_phase_mesh=True,
		title=f"Full ternary boundary plot ({'-'.join(elements)})",
	)

	fig.write_html(str(OUTPUT_PLOT_PATH))
	print("=" * 72)
	print("TERNARY WORKFLOW COMPLETE")
	print("=" * 72)
	print(f"System: {'-'.join(elements)}")
	print(f"Include polymorphs: {INCLUDE_POLYMORPHS}")
	print(f"Include solid solutions: {INCLUDE_SOLID_SOLUTIONS}")
	print(f"Interpolation scheme: {INTERPOLATION_SCHEME}")
	print(f"Keep single-phase liquid simplices: {KEEP_SINGLE_PHASE_LIQUID_SIMPLICES}")
	print(f"Validated all temperature slices: {VALIDATE_ALL_TEMPERATURE_SLICES}")
	print(f"Saved full ternary plot: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
	main()

