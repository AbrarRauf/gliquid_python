"""HPC runner for General HSX temperature-block sweeps.

This script is intended for array-job execution where each task receives a
temperature block through environment variables, then runs interpolation and
equilibrium solve for that explicit block and writes lower-hull cache artifacts.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from general_HSX import GeneralEquilibrium, GeneralInterpolation


def _parse_bool_env(name: str, default: bool) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return bool(default)
	val = raw.strip().lower()
	if val in {"1", "true", "yes", "y", "on"}:
		return True
	if val in {"0", "false", "no", "n", "off"}:
		return False
	raise ValueError(f"Invalid boolean for {name}: {raw}")


def _parse_float_env(name: str, default: float) -> float:
	raw = os.getenv(name)
	if raw is None:
		return float(default)
	return float(raw)


def _parse_int_env(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return int(default)
	return int(raw)


def _parse_system_elements(raw: str) -> List[str]:
	elems = [tok.strip() for tok in raw.split(",") if tok.strip()]
	if len(elems) < 3:
		raise ValueError(
			f"SYSTEM_ELEMENTS must contain at least 3 elements, got: {raw}"
		)
	return elems


def _parse_temp_block_k(raw: str) -> Tuple[float, float]:
	tokens = [t for t in re.split(r"[,;\s]+", raw.strip()) if t]
	if len(tokens) != 2:
		raise ValueError(
			f"TEMP_BLOCK_K must have exactly two numbers, got: {raw}"
		)
	tmin = float(tokens[0])
	tmax = float(tokens[1])
	if tmax < tmin:
		raise ValueError(f"TEMP_BLOCK_K has Tmax < Tmin: {raw}")
	return float(tmin), float(tmax)


def _load_binary_params_from_file(df: pd.DataFrame, pair_label: str) -> List[float]:
	"""Load canonical [L0_a, L0_b, L1_a, L1_b] for one binary pair.

	If the file only contains reversed pair ordering, L1 terms are sign-flipped
	to map into canonical sorted pair convention.
	"""
	canonical = "-".join(sorted(pair_label.split("-")))
	p0, p1 = canonical.split("-")
	reversed_label = f"{p1}-{p0}"

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

	raise ValueError(f"Binary pair '{pair_label}' not found in parameter file.")


def main() -> None:
	repo_root = Path(__file__).resolve().parents[2]

	system_raw = os.getenv("SYSTEM_ELEMENTS")
	if not system_raw:
		raise ValueError("SYSTEM_ELEMENTS is required, example: Al,Hf,Nb,W")
	temp_block_raw = os.getenv("TEMP_BLOCK_K")
	if not temp_block_raw:
		raise ValueError("TEMP_BLOCK_K is required, example: 733.5,833.5")

	elements = _parse_system_elements(system_raw)
	tmin_k, tmax_k = _parse_temp_block_k(temp_block_raw)

	param_xlsx_path = Path(
		os.getenv(
			"PARAM_XLSX_PATH",
			str(repo_root / "data" / "high_component" / "ssol_fits_linear_model_legacy_refs-tau_penalty.xlsx"),
		)
	)
	output_root = Path(os.getenv("OUTPUT_ROOT", str(repo_root / "all_dumps" / "hpc_runs")))

	grid_delta = _parse_float_env("GRID_DELTA", 0.025)
	temp_delta_k = _parse_float_env("TEMP_DELTA_K", 5.0)
	include_ref_ss = _parse_bool_env("INCLUDE_REF_SS", True)
	include_polymorphs = _parse_bool_env("INCLUDE_POLYMORPHS", False)
	use_mp_cache = _parse_bool_env("USE_MP_CACHE", False)
	vertical_simplices = _parse_bool_env("VERTICAL_SIMPLICES", False)
	eq_progress_every = _parse_int_env("EQ_PROGRESS_EVERY", 1)

	ref_ss_omegas_path_env = os.getenv("REF_SS_OMEGAS_PATH")
	if ref_ss_omegas_path_env:
		ref_ss_omegas_path = str(Path(ref_ss_omegas_path_env).resolve())
	else:
		candidate = repo_root / "matrix_data" / "omegas.json"
		ref_ss_omegas_path = str(candidate) if candidate.exists() else None

	if include_ref_ss and not ref_ss_omegas_path:
		raise FileNotFoundError(
			"include_ref_ss=True but no omegas path was provided and default "
			"matrix_data/omegas.json was not found. Set REF_SS_OMEGAS_PATH."
		)

	canonical_elements = sorted(elements)
	system_name = "-".join(canonical_elements)
	block_label = f"T{tmin_k:.2f}_{tmax_k:.2f}"
	run_dir = output_root / system_name / block_label
	run_dir.mkdir(parents=True, exist_ok=True)

	print("=" * 72)
	print("GENERAL HSX HPC BLOCK RUNNER")
	print("=" * 72)
	print(f"System input: {elements}")
	print(f"Canonical system: {system_name}")
	print(f"Temperature block (K): ({tmin_k:.2f}, {tmax_k:.2f})")
	print(f"Grid delta: {grid_delta}")
	print(f"Temperature delta (K): {temp_delta_k}")
	print(f"Include ref solid solutions: {include_ref_ss}")
	print(f"Include polymorphs: {include_polymorphs}")
	print(f"Use MP cache: {use_mp_cache}")
	print(f"Vertical simplices: {vertical_simplices}")
	print(f"Parameter file: {param_xlsx_path}")
	print(f"Output run dir: {run_dir}")

	if not param_xlsx_path.exists():
		raise FileNotFoundError(f"Parameter file not found: {param_xlsx_path}")

	binary_pairs = [
		f"{canonical_elements[i]}-{canonical_elements[j]}"
		for i, j in combinations(range(len(canonical_elements)), 2)
	]
	binary_param_df = pd.read_excel(param_xlsx_path)
	binary_L_dict: Dict[str, List[float]] = {
		pair: _load_binary_params_from_file(binary_param_df, pair)
		for pair in binary_pairs
	}

	interp = GeneralInterpolation(
		elements=elements,
		output_dir=str(run_dir),
		grid_delta=grid_delta,
		temp_delta_k=temp_delta_k,
		temp_bounds_k=(tmin_k, tmax_k),
		include_polymorphs=include_polymorphs,
		include_ref_solid_solutions=include_ref_ss,
		ref_solid_solutions_path=ref_ss_omegas_path,
	)
	interp.set_binary_params(binary_L_dict)

	print("\n[1/2] Running interpolation...")
	interp.interpolate(use_mp_cache=use_mp_cache)

	print("\n[2/2] Running equilibrium lower-hull solve...")
	eq_solver = GeneralEquilibrium(interp.gtx_data)
	eq_df = eq_solver.solve(
		vertical_simplices=vertical_simplices,
		print_progress=True,
		progress_every=eq_progress_every,
	)

	lower_hull_cache_dir = run_dir / "lower_hull_cache"
	lower_hull_cache_dir.mkdir(parents=True, exist_ok=True)
	cache_name = f"{system_name}_{block_label}"
	cache_paths = eq_solver.save_lower_hull_cache(
		cache_dir=str(lower_hull_cache_dir),
		cache_name=cache_name,
		include_gtx=True,
		vertical_simplices=vertical_simplices,
	)

	progress = eq_solver.get_temperature_progress()
	summary = {
		"timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		"system_input": elements,
		"system_canonical": canonical_elements,
		"temp_block_k": [tmin_k, tmax_k],
		"grid_delta": grid_delta,
		"temp_delta_k": temp_delta_k,
		"include_ref_ss": include_ref_ss,
		"include_polymorphs": include_polymorphs,
		"use_mp_cache": use_mp_cache,
		"vertical_simplices": vertical_simplices,
		"n_gtx_rows": int(len(interp.gtx_data) if interp.gtx_data is not None else 0),
		"n_equilibrium_rows": int(len(eq_df)),
		"n_simplex_rows": int(len(eq_solver.simplex_df) if eq_solver.simplex_df is not None else 0),
		"temperature_progress": progress,
		"cache_paths": cache_paths,
	}

	summary_path = run_dir / f"run_summary_{block_label}.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	snapshot_rows = int(_parse_int_env("SNAPSHOT_ROWS", 10000))
	if interp.gtx_data is not None and snapshot_rows > 0:
		snapshot_path = run_dir / f"gtx_snapshot_{block_label}.csv"
		interp.gtx_data.head(snapshot_rows).to_csv(snapshot_path, index=False)
		print(f"Saved GTX snapshot: {snapshot_path}")

	print("\nRun complete.")
	print(f"GTX rows: {summary['n_gtx_rows']}")
	print(f"Equilibrium rows: {summary['n_equilibrium_rows']}")
	print(f"Simplex rows: {summary['n_simplex_rows']}")
	print(f"Lower-hull cache manifest: {cache_paths['manifest']}")
	print(f"Run summary: {summary_path}")


if __name__ == "__main__":
	main()
