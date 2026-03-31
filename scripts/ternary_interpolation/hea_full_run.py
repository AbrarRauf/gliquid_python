"""End-to-end postprocess + visualization runner for 5-component HEA systems.

Workflow:
1) main_post(): stitch lower-hull blocks, extract/save phase-boundary dataframe,
   and write eutectic + temperature-boundary diagnostics summary.
2) main_viz(): load saved phase-boundary dataframe and generate batches of
   binary/ternary/quaternary slice HTML plots.
"""

from __future__ import annotations

import json
import os
import re
from math import comb
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from general_plotter import GeneralPostProcess, PhaseBoundaryPlotter


def _infer_comp_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def _nearest_available(series: pd.Series, target: float) -> float:
    vals = np.sort(series.dropna().unique().astype(float))
    if len(vals) == 0:
        return float(target)
    idx = int(np.argmin(np.abs(vals - float(target))))
    return float(vals[idx])


def _is_liquid_phase(phase: str) -> bool:
    return str(phase).strip().upper() in {"L", "LIQUID"}


def _is_solution_phase(phase: str) -> bool:
    p = str(phase).upper()
    if _is_liquid_phase(p):
        return False
    return re.search(r"(FCC|BCC|HCP|A1|A2|A3|A4|SOLUTION|_SS|\bSS\b)", p) is not None


def _full_named_df(df: pd.DataFrame, comp_cols: Sequence[str], element_names: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["x_dep"] = 1.0 - out[list(comp_cols)].sum(axis=1)
    full_cols = ["x_dep"] + list(comp_cols)
    if len(full_cols) != len(element_names):
        raise ValueError(f"Component/element length mismatch: {len(full_cols)} vs {len(element_names)}")
    for src, dst in zip(full_cols, element_names):
        out[dst] = out[src].astype(float)
    return out


def _composition_dict(row: pd.Series, element_names: Sequence[str], precision: int = 4) -> Dict[str, float]:
    return {f"x_{el}": round(float(row[el]), precision) for el in element_names}


def _composition_key_from_row(row: pd.Series, element_names: Sequence[str], ndigits: int = 8) -> Tuple[float, ...]:
    return tuple(round(float(row[el]), ndigits) for el in element_names)


def _composition_key_grid(row: pd.Series, element_names: Sequence[str], grid_delta: float) -> Tuple[int, ...]:
    if grid_delta <= 0.0:
        return tuple(int(round(float(row[el]) * 1e8)) for el in element_names)
    return tuple(int(round(float(row[el]) / grid_delta)) for el in element_names)


def _save_boundary_csv_snapshot(
    df: pd.DataFrame,
    output_dir: Path,
    system_name: str,
    max_rows: int = 10000,
    dump_full: bool = False,
    random_state: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    t_min = float(df["T_K"].min()) if "T_K" in df.columns and not df.empty else 0.0
    t_max = float(df["T_K"].max()) if "T_K" in df.columns and not df.empty else 0.0

    if dump_full or len(df) <= max_rows or "Phase" not in df.columns or "T_K" not in df.columns:
        out_df = df.copy().reset_index(drop=True)
        name = f"{system_name}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium_full.csv" if dump_full else f"{system_name}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium_sample.csv"
    else:
        work = df.copy()
        n_unique_t = int(work["T_K"].nunique())
        q = int(max(2, min(24, n_unique_t))) if n_unique_t > 0 else 2
        try:
            work["_temp_bin"] = pd.qcut(work["T_K"].astype(float), q=q, duplicates="drop")
        except Exception:
            work["_temp_bin"] = "all"
        groups = work.groupby(["Phase", "_temp_bin"], observed=True, sort=False)
        n_groups = max(1, int(groups.ngroups))
        per_group = max(1, int(max_rows) // n_groups)

        parts = []
        for _, g in groups:
            parts.append(g.sample(n=min(len(g), per_group), random_state=random_state))
        out_df = pd.concat(parts, axis=0) if parts else work.iloc[0:0].copy()

        if len(out_df) < max_rows:
            rem = work.drop(index=out_df.index, errors="ignore")
            need = int(max_rows) - len(out_df)
            if need > 0 and len(rem) > 0:
                out_df = pd.concat([out_df, rem.sample(n=min(need, len(rem)), random_state=random_state)], axis=0)
        if len(out_df) > max_rows:
            out_df = out_df.sample(n=int(max_rows), random_state=random_state)
        out_df = out_df.drop(columns=["_temp_bin"], errors="ignore").sort_values(by=["Phase", "T_K"]).reset_index(drop=True)
        name = f"{system_name}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium_sample.csv"

    out_path = output_dir / name
    out_df.to_csv(out_path, index=False)
    return out_path


def _compute_hea_summary(
    equilibrium_df: pd.DataFrame,
    element_names: Sequence[str],
    hea_min_fraction: float = 0.10,
    grid_delta: float = 0.05,
) -> Dict[str, object]:
    eq = equilibrium_df.copy()
    if eq.empty:
        return {
            "hea_filter": {"min_element_fraction": float(hea_min_fraction), "n_rows_in_scope": 0},
            "hea_eutectic": None,
            "phases_at_tmin_not_tmax": [],
            "high_t_liquid_coverage": {"complete": None, "message": "No data available"},
            "solution_phase_highest_melting_with_liquid_boundary": {},
            "intermetallic_highest_melting_with_liquid_boundary": {},
            "warnings": {
                "liquid_present_at_tmin": None,
                "incomplete_liquid_coverage_high_t": None,
            },
        }

    comp_cols = _infer_comp_cols(eq)
    eq_full = _full_named_df(eq, comp_cols=comp_cols, element_names=element_names)
    hea_mask = np.ones(len(eq_full), dtype=bool)
    for el in element_names:
        hea_mask &= eq_full[el].to_numpy(dtype=float) >= float(hea_min_fraction) - 1e-12
    eq_hea = eq_full.loc[hea_mask].copy().reset_index(drop=True)

    # Build simplex -> phase-set map once (for coexistence reporting).
    simplex_phase_sets = (
        eq.groupby("simplex_id")["Phase"]
        .apply(lambda s: {str(v) for v in s.tolist()})
        .to_dict()
    ) if "simplex_id" in eq.columns else {}

    # 1) HEA eutectic from liquid low-T per composition, then HEA composition filter.
    hea_eutectic = None
    liquid = eq_full[eq_full["Phase"].astype(str).apply(_is_liquid_phase)].copy()
    liq_min_rows = pd.DataFrame()
    if not liquid.empty:
        liquid = liquid.copy()
        liquid["_ckey"] = liquid.apply(lambda r: _composition_key_grid(r, element_names, grid_delta), axis=1)
        idx_min = liquid.groupby("_ckey")["T_K"].idxmin()
        liq_min_rows = liquid.loc[idx_min].copy().reset_index(drop=True)

    if not liq_min_rows.empty:
        hea_liq = liq_min_rows.copy()
        for el in element_names:
            hea_liq = hea_liq[hea_liq[el].to_numpy(dtype=float) >= float(hea_min_fraction) - 1e-12]
        if not hea_liq.empty:
            rep = hea_liq.sort_values(by=["T_K", "G" if "G" in hea_liq.columns else "T_K"], ascending=[True, True]).iloc[0]
            sid = int(rep["simplex_id"]) if "simplex_id" in rep and pd.notna(rep["simplex_id"]) else None
            phases = sorted(simplex_phase_sets.get(sid, set())) if sid is not None else []
            hea_eutectic = {
                "temperature_K": float(rep["T_K"]),
                "temperature_C": float(rep["T_K"] - 273.15),
                "composition": _composition_dict(rep, element_names=element_names, precision=4),
                "simplex_id": sid,
                "phases_in_simplex": phases,
                "coexisting_solids": [p for p in phases if not _is_liquid_phase(p)],
            }

    # 2) Phases at t_min but not t_max (HEA scope), and liquid coverage metric.
    if not eq_hea.empty:
        tmin = float(eq_hea["T_K"].min())
        tmax = float(eq_hea["T_K"].max())
        p_tmin = set(eq_hea[np.isclose(eq_hea["T_K"].to_numpy(dtype=float), tmin, atol=1e-9, rtol=0.0)]["Phase"].astype(str).tolist())
        p_tmax = set(eq_hea[np.isclose(eq_hea["T_K"].to_numpy(dtype=float), tmax, atol=1e-9, rtol=0.0)]["Phase"].astype(str).tolist())
        phases_tmin_not_tmax = sorted(p_tmin - p_tmax)
        liquid_present_tmin = any(_is_liquid_phase(p) for p in p_tmin)
    else:
        tmin = None
        tmax = None
        p_tmin = set()
        p_tmax = set()
        phases_tmin_not_tmax = []
        liquid_present_tmin = None

    liquid_cov = {
        "complete": None,
        "message": "No data available",
        "coverage_percent": None,
        "n_grid_points_total": 0,
        "n_compositions_with_liquid_min_t": 0,
    }
    if not liq_min_rows.empty:
        n_comp = len(element_names)
        n_steps = int(round(1.0 / grid_delta)) if grid_delta > 0 else 0
        total_grid = comb(n_steps + n_comp - 1, n_comp - 1) if (grid_delta > 0 and abs(n_steps * grid_delta - 1.0) < 1e-8) else int(liq_min_rows["_ckey"].nunique())
        n_liq = int(liq_min_rows["_ckey"].nunique())
        cov_pct = (100.0 * n_liq / float(total_grid)) if total_grid > 0 else None
        complete = bool((cov_pct is not None) and (cov_pct > 99.0))
        liquid_cov = {
            "complete": complete,
            "message": "complete liquidus coverage at high T" if complete else "incomplete liquid coverage at high T",
            "coverage_percent": cov_pct,
            "n_grid_points_total": int(total_grid),
            "n_compositions_with_liquid_min_t": int(n_liq),
        }

    # 3) Highest T for SS phases by per-composition Tmax; intermetallic highest T overall.
    phases_all = sorted({str(p) for p in eq["Phase"].astype(str).tolist()})
    solution_phases = [p for p in phases_all if (not _is_liquid_phase(p)) and _is_solution_phase(p)]
    intermetallic_phases = [p for p in phases_all if (not _is_liquid_phase(p)) and (not _is_solution_phase(p))]

    solution_stats: Dict[str, Dict[str, object]] = {}
    for phase in solution_phases:
        if eq_hea.empty:
            continue
        cand = eq_hea[eq_hea["Phase"].astype(str) == phase].copy()
        if cand.empty:
            continue
        cand["_ckey"] = cand.apply(lambda r: _composition_key_grid(r, element_names, grid_delta), axis=1)
        idx_max = cand.groupby("_ckey")["T_K"].idxmax()
        rep = cand.loc[idx_max].sort_values(by=["T_K", "G" if "G" in cand.columns else "T_K"], ascending=[False, True]).iloc[0]
        solution_stats[phase] = {
            "highest_melting_T_K": float(rep["T_K"]),
            "highest_melting_T_C": float(rep["T_K"] - 273.15),
            "composition": _composition_dict(rep, element_names=element_names, precision=4),
            "simplex_id": int(rep["simplex_id"]) if "simplex_id" in rep and pd.notna(rep["simplex_id"]) else None,
        }

    intermetallic_stats: Dict[str, Dict[str, object]] = {}
    for phase in intermetallic_phases:
        cand = eq_full[eq_full["Phase"].astype(str) == phase].copy()
        if cand.empty:
            continue
        rep = cand.sort_values(by=["T_K", "G"], ascending=[False, True]).iloc[0]
        sid = int(rep["simplex_id"]) if "simplex_id" in rep and pd.notna(rep["simplex_id"]) else None
        phases = simplex_phase_sets.get(sid, set()) if sid is not None else set()
        if not any(_is_liquid_phase(p) for p in phases):
            continue
        intermetallic_stats[phase] = {
            "highest_melting_T_K": float(rep["T_K"]),
            "highest_melting_T_C": float(rep["T_K"] - 273.15),
            "simplex_id": sid,
            "coexists_with_liquid": True,
        }

    warn_incomplete_liq_cov = None if liquid_cov.get("complete") is None else (not bool(liquid_cov["complete"]))

    return {
        "hea_filter": {
            "min_element_fraction": float(hea_min_fraction),
            "n_rows_in_scope": int(len(eq_hea)),
        },
        "hea_eutectic": hea_eutectic,
        "high_t_liquid_coverage": liquid_cov,
        "solution_phase_highest_melting_with_liquid_boundary": solution_stats,
        "intermetallic_highest_melting_with_liquid_boundary": intermetallic_stats,
        "warnings": {
            "liquid_present_at_tmin": liquid_present_tmin,
            "incomplete_liquid_coverage_high_t": warn_incomplete_liq_cov,
        },
    }


def _reduce_to_quaternary_df(
    df_full_named: pd.DataFrame,
    selected_full_components: Sequence[str],
) -> pd.DataFrame:
    """Project a filtered 5-component dataframe onto a 4-component quaternary frame.

    selected_full_components defines the quaternary composition basis in order:
    first entry is dependent component, remaining 3 map to x0,x1,x2.
    """
    if len(selected_full_components) != 4:
        raise ValueError("selected_full_components must contain exactly 4 component names.")

    dep = selected_full_components[0]
    x0n = selected_full_components[1]
    x1n = selected_full_components[2]
    x2n = selected_full_components[3]

    required = [dep, x0n, x1n, x2n, "Phase", "T_K"]
    missing = [c for c in required if c not in df_full_named.columns]
    if missing:
        raise ValueError(f"Missing required columns for quaternary projection: {missing}")

    out = pd.DataFrame()
    out["x0"] = df_full_named[x0n].astype(float)
    out["x1"] = df_full_named[x1n].astype(float)
    out["x2"] = df_full_named[x2n].astype(float)
    out["Phase"] = df_full_named["Phase"].astype(str)
    out["T_K"] = df_full_named["T_K"].astype(float)

    if "T_C" in df_full_named.columns:
        out["T_C"] = df_full_named["T_C"].astype(float)
    else:
        out["T_C"] = out["T_K"] - 273.15

    if "G" in df_full_named.columns:
        out["G"] = df_full_named["G"].astype(float)
    if "simplex_id" in df_full_named.columns:
        out["simplex_id"] = df_full_named["simplex_id"]
    if "source_row_id" in df_full_named.columns:
        out["source_row_id"] = df_full_named["source_row_id"]

    # Keep only physically valid projected points.
    x_dep = 1.0 - (out["x0"] + out["x1"] + out["x2"])
    valid = (x_dep >= -1e-9) & (x_dep <= 1.0 + 1e-9)
    out = out.loc[valid].copy().reset_index(drop=True)
    return out


def main_post() -> None:
    system_name = "Al-Hf-Nb-W-Zr"
    element_names = ["Al", "Hf", "Nb", "W", "Zr"]
    system_cache_dir = Path(
        os.getenv(
            "HEA_SYSTEM_CACHE_DIR",
            "all_dumps/quaternary_demo/lower_hull_cache/Al-Hf-Nb-W-Zr",
        )
    )
    plotter_dir = Path(os.getenv("HEA_PLOTTER_DIR", "all_dumps/quaternary_demo/plotter_dir"))
    dump_full_csv = os.getenv("HEA_DUMP_FULL_EQUILIBRIUM_CSV", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    sample_max_rows = int(os.getenv("HEA_SAMPLE_MAX_ROWS", "10000"))
    hea_min_fraction = float(os.getenv("HEA_MIN_ELEMENT_FRACTION", "0.15"))
    use_existing_boundary = os.getenv("HEA_USE_EXISTING_BOUNDARY_FILE", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    grid_delta = float(os.getenv("HEA_GRID_DELTA", "0.05"))

    print("=" * 80)
    print("HEA MAIN_POST")
    print("=" * 80)
    print(f"System cache dir: {system_cache_dir}")
    print(f"Plotter dir: {plotter_dir}")
    print(f"Dump full equilibrium CSV: {dump_full_csv}")
    print(f"Sample max rows (when not full): {sample_max_rows}")
    print(f"HEA minimum element fraction: {hea_min_fraction}")
    print(f"Use existing boundary file: {use_existing_boundary}")
    print(f"Grid delta: {grid_delta}")

    plotter_dir.mkdir(parents=True, exist_ok=True)
    phase_boundary_path = None
    phase_boundary_df = None

    if use_existing_boundary:
        explicit_file = os.getenv("HEA_PHASE_BOUNDARY_FILE")
        if explicit_file:
            candidate = Path(explicit_file)
            if candidate.exists():
                phase_boundary_path = candidate
        if phase_boundary_path is None:
            candidates = sorted(plotter_dir.glob(f"{system_name}_T*_phase_boundary_equilibrium.pkl.gz"))
            if candidates:
                phase_boundary_path = candidates[-1]
        if phase_boundary_path is not None and phase_boundary_path.exists():
            phase_boundary_df = pd.read_pickle(phase_boundary_path, compression="gzip")

    if phase_boundary_df is None:
        post = GeneralPostProcess(
            system_cache_dir=str(system_cache_dir),
            recursive=True,
            load_gtx=True,
        )
        post.load_and_stitch()
        post.extract_phase_boundary_equilibrium(min_unique_phases=2)
        phase_boundary_path = post.save_phase_boundary_equilibrium(output_dir=str(plotter_dir))
        phase_boundary_df = post.phase_boundary_equilibrium_df.copy()

    phase_boundary_sample_csv = _save_boundary_csv_snapshot(
        df=phase_boundary_df,
        output_dir=plotter_dir,
        system_name=system_name,
        max_rows=sample_max_rows,
        dump_full=dump_full_csv,
    )

    complete_summary = _compute_hea_summary(
        equilibrium_df=phase_boundary_df,
        element_names=element_names,
        hea_min_fraction=hea_min_fraction,
        grid_delta=grid_delta,
    )

    summary = {
        "system": system_name,
        "system_cache_dir": str(system_cache_dir),
        "phase_boundary_file": str(phase_boundary_path),
        "phase_boundary_sample_csv": str(phase_boundary_sample_csv),
        "phase_boundary_csv_mode": "full" if dump_full_csv else "sample",
        "post_source": "existing_boundary_file" if use_existing_boundary else "stitched_from_blocks",
        "complete_summary": complete_summary,
    }

    summary_path = plotter_dir / f"{system_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved phase-boundary file: {phase_boundary_path}")
    if dump_full_csv:
        print(f"Saved phase-boundary full CSV: {phase_boundary_sample_csv}")
    else:
        print(f"Saved phase-boundary sample CSV: {phase_boundary_sample_csv}")
    print(f"Saved complete summary: {summary_path}")
    w = complete_summary.get("warnings", {}) if isinstance(complete_summary, dict) else {}
    if w.get("liquid_present_at_tmin") is True:
        print("[WARN] L phase present at t_min within HEA composition scope.")
    if w.get("incomplete_liquid_coverage_high_t") is True:
        print("[WARN] Incomplete liquid coverage at high T within HEA composition scope.")


def main_viz() -> None:
    system_name = "Al-Hf-Nb-W-Zr"
    element_names = ["Al", "Hf", "Nb", "W", "Zr"]
    grid_delta = float(os.getenv("HEA_GRID_DELTA", "0.05"))
    tol = float(os.getenv("HEA_SLICE_TOL", str(max(0.5 * grid_delta, 1e-6))))
    use_phase_extrema = os.getenv("HEA_SLICE_PHASE_EXTREMA", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    use_ternary_phase_mesh = os.getenv("HEA_TERNARY_PHASE_MESH", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    ss_cluster_factor = float(os.getenv("HEA_TERNARY_SS_CLUSTER_FACTOR", "1.75"))
    if not use_phase_extrema:
        use_ternary_phase_mesh = False

    plotter_dir = Path(os.getenv("HEA_PLOTTER_DIR", "all_dumps/quaternary_demo/plotter_dir"))
    explicit_file = os.getenv("HEA_PHASE_BOUNDARY_FILE")
    if explicit_file:
        phase_boundary_file = Path(explicit_file)
    else:
        candidates = sorted(plotter_dir.glob(f"{system_name}_T*_phase_boundary_equilibrium.pkl.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"No phase-boundary file found for {system_name} in {plotter_dir}. Run main_post() first."
            )
        phase_boundary_file = candidates[-1]

    html_out_dir = plotter_dir / "hea_dump_main"
    html_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HEA MAIN_VIZ")
    print("=" * 80)
    print(f"Phase boundary file: {phase_boundary_file}")
    print(f"HTML output dir: {html_out_dir}")
    print(f"Grid delta: {grid_delta}")
    print(f"Slice tolerance: {tol}")
    print(f"Slice phase-extrema filter: {use_phase_extrema}")
    print(f"Ternary phase mesh enabled: {use_ternary_phase_mesh}")
    print(f"Ternary SS cluster factor: {ss_cluster_factor}")

    df = pd.read_pickle(phase_boundary_file, compression="gzip")
    comp_cols = _infer_comp_cols(df)
    plotter = PhaseBoundaryPlotter(
        equilibrium_df=df,
        composition_cols=comp_cols,
        element_names=element_names,
    )
    full_df = plotter._full_component_df()
    full_names = plotter._full_component_names()

    if len(full_names) != 5:
        raise ValueError(f"Expected 5 total components for HEA run, got {len(full_names)}: {full_names}")

    # -----------------------------------------------------------------------------
    # Hard-coded example slices (intentionally chosen as random-looking demos).
    # These are explicit so users can copy/modify them directly.
    #
    # How to use:
    # - Binary: pick (comp_a, comp_b), then set fixed values for the other 3 comps.
    # - Ternary: pick (comp_a, comp_b, comp_c), then set fixed values for the other 2 comps.
    # - Quaternary: pick 4 components, fix the omitted 1 component, then project with
    #   _reduce_to_quaternary_df and plot with plot_quaternary_phase_tetrahedral.
    # -----------------------------------------------------------------------------
    print("Generating hard-coded random-example demos only (3 binary, 3 ternary, 3 quaternary).")

    def _snap_fixed(fixed_components: Dict[str, float]) -> Dict[str, float]:
        snapped: Dict[str, float] = {}
        for name, val in fixed_components.items():
            snapped[name] = _nearest_available(full_df[name], float(val))
        return snapped

    # ------------------ Binary examples (3) --------------------------------------
    binary_examples = [
        {
            "comp_a": "Al",
            "comp_b": "Zr",
            "fixed": {"Hf": 0.15, "Nb": 0.20, "W": 0.10},
        },
        {
            "comp_a": "Al",
            "comp_b": "Zr",
            "fixed": {"Hf": 0.10, "Nb": 0.15, "W": 0.25},
        },
        {
            "comp_a": "Al",
            "comp_b": "Zr",
            "fixed": {"Hf": 0.20, "Nb": 0.10, "W": 0.15},
        },
    ]

    bin_saved = 0
    for i, ex in enumerate(binary_examples, start=1):
        fixed = _snap_fixed(ex["fixed"])
        try:
            fig = plotter.plot_binary_slice_tx(
                comp_a=ex["comp_a"],
                comp_b=ex["comp_b"],
                fixed_components=fixed,
                tolerance=tol,
                phase_extrema_filter=use_phase_extrema,
                title=f"RANDOM-EXAMPLE binary {ex['comp_a']}-{ex['comp_b']} | fixed {fixed}",
            )
        except Exception as e:
            print(f"[SKIP][binary example {i}] {ex['comp_a']}-{ex['comp_b']}: {e}")
            continue
        out = html_out_dir / f"binary_example_{i:02d}_{ex['comp_a']}_{ex['comp_b']}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        bin_saved += 1

    # ------------------ Ternary examples (3) -------------------------------------
    ternary_examples = [
        {
            "comp_a": "Al",
            "comp_b": "Nb",
            "comp_c": "Zr",
            "fixed": {"Hf": 0.15, "W": 0.15},
        },
        {
            "comp_a": "Al",
            "comp_b": "Nb",
            "comp_c": "Zr",
            "fixed": {"Hf": 0.20, "W": 0.10},
        },
        {
            "comp_a": "Al",
            "comp_b": "Nb",
            "comp_c": "Zr",
            "fixed": {"Hf": 0.10, "W": 0.20},
        },
    ]

    tern_saved = 0
    for i, ex in enumerate(ternary_examples, start=1):
        fixed = _snap_fixed(ex["fixed"])
        try:
            fig = plotter.plot_ternary_slice(
                comp_a=ex["comp_a"],
                comp_b=ex["comp_b"],
                comp_c=ex["comp_c"],
                fixed_components=fixed,
                tolerance=tol,
                phase_extrema_filter=use_phase_extrema,
                ternary_phase_mesh=use_ternary_phase_mesh,
                slice_grid_delta=grid_delta,
                ss_cluster_factor=ss_cluster_factor,
                title=f"RANDOM-EXAMPLE ternary {ex['comp_a']}-{ex['comp_b']}-{ex['comp_c']} | fixed {fixed}",
                color_by="Phase",
            )
        except Exception as e:
            print(f"[SKIP][ternary example {i}] {ex['comp_a']}-{ex['comp_b']}-{ex['comp_c']}: {e}")
            continue
        out = html_out_dir / f"ternary_example_{i:02d}_{ex['comp_a']}_{ex['comp_b']}_{ex['comp_c']}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        tern_saved += 1

    # ------------------ Quaternary examples (3) ----------------------------------
    quaternary_examples = [
        {
            "selected4": ["Al", "Hf", "Nb", "Zr"],
            "fixed": {"W": 0.10},
            "phase_filter": "L",
            "temperature_extrema": "min",
        },
        {
            "selected4": ["Al", "Hf", "Nb", "Zr"],
            "fixed": {"W": 0.15},
            "phase_filter": "L",
            "temperature_extrema": "min",
        },
        {
            "selected4": ["Al", "Hf", "Nb", "Zr"],
            "fixed": {"W": 0.20},
            "phase_filter": "L",
            "temperature_extrema": "min",
        },
    ]

    quat_saved = 0
    for i, ex in enumerate(quaternary_examples, start=1):
        selected4 = ex["selected4"]
        fixed = _snap_fixed(ex["fixed"])
        try:
            df_slice = plotter.filter_by_fixed_components(fixed_components=fixed, tolerance=tol)
            df_q = _reduce_to_quaternary_df(df_slice, selected_full_components=selected4)
            if df_q.empty:
                raise ValueError("Projected quaternary slice is empty")
            qplotter = PhaseBoundaryPlotter(
                equilibrium_df=df_q,
                composition_cols=["x0", "x1", "x2"],
                element_names=selected4,
            )
            fixed_name = next(iter(fixed.keys()))
            fig = qplotter.plot_quaternary_phase_tetrahedral(
                phase_filter=str(ex["phase_filter"]),
                temperature_extrema=str(ex["temperature_extrema"]),
                composition_tol=tol,
                title=(
                    f"RANDOM-EXAMPLE quaternary {tuple(selected4)} | "
                    f"fixed {fixed_name}={fixed[fixed_name]:.2f}"
                ),
            )
        except Exception as e:
            print(f"[SKIP][quaternary example {i}] {selected4}: {e}")
            continue

        out = html_out_dir / f"quaternary_example_{i:02d}_{'_'.join(selected4)}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        quat_saved += 1

    run_summary = {
        "system": system_name,
        "phase_boundary_file": str(phase_boundary_file),
        "html_out_dir": str(html_out_dir),
        "grid_delta": grid_delta,
        "slice_tolerance": tol,
        "phase_extrema_filter": use_phase_extrema,
        "saved_counts": {
            "binary_html": int(bin_saved),
            "ternary_html": int(tern_saved),
            "quaternary_html": int(quat_saved),
        },
    }
    summary_path = html_out_dir / "hea_viz_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Saved HEA viz summary: {summary_path}")
    print(f"Saved binary HTML count: {bin_saved}")
    print(f"Saved ternary HTML count: {tern_saved}")
    print(f"Saved quaternary HTML count: {quat_saved}")


if __name__ == "__main__":
    # mode = os.getenv("HEA_MAIN", "post").strip().lower()
    mode = os.getenv("HEA_MAIN", "viz").strip().lower()
    if mode == "post":
        main_post()
    elif mode == "viz":
        main_viz()
    else:
        raise ValueError(f"Invalid HEA_MAIN value: {mode}. Use 'post' or 'viz'.")
