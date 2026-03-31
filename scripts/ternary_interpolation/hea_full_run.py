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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from general_plotter import GeneralHSXPostprocess, PhaseBoundaryPlotter


def _infer_comp_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def _nearest_available(series: pd.Series, target: float) -> float:
    vals = np.sort(series.dropna().unique().astype(float))
    if len(vals) == 0:
        return float(target)
    idx = int(np.argmin(np.abs(vals - float(target))))
    return float(vals[idx])


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
    system_cache_dir = Path(
        os.getenv(
            "HEA_SYSTEM_CACHE_DIR",
            "all_dumps/quaternary_demo/lower_hull_cache/Al-Hf-Nb-W-Zr",
        )
    )
    plotter_dir = Path(os.getenv("HEA_PLOTTER_DIR", "all_dumps/quaternary_demo/plotter_dir"))

    print("=" * 80)
    print("HEA MAIN_POST")
    print("=" * 80)
    print(f"System cache dir: {system_cache_dir}")
    print(f"Plotter dir: {plotter_dir}")

    post = GeneralHSXPostprocess(
        system_cache_dir=str(system_cache_dir),
        recursive=True,
        load_gtx=True,
    )
    post.load_and_stitch()
    post.extract_phase_boundary_equilibrium(min_unique_phases=2)
    phase_boundary_path = post.save_phase_boundary_equilibrium(output_dir=str(plotter_dir))

    temp_diag = post.evaluate_tmax_filter_change(atol=1e-9)
    warn_tmax = bool(temp_diag.get("tmax_changed_after_filter") is False)
    warn_tmin = bool(temp_diag.get("liquid_present_at_full_tmin") is True)

    summary = {
        "system": system_name,
        "system_cache_dir": str(system_cache_dir),
        "phase_boundary_file": str(phase_boundary_path),
        "global_lowest_eutectic": post.global_eutectic,
        "temperature_boundary_diagnostic": temp_diag,
        "warnings": {
            "tmax_filter_unchanged": warn_tmax,
            "tmin_contains_liquid": warn_tmin,
        },
    }

    plotter_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plotter_dir / f"{system_name}_eutectic_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved phase-boundary file: {phase_boundary_path}")
    print(f"Saved eutectic summary: {summary_path}")
    if warn_tmax:
        print("[WARN] Tmax filter unchanged; consider extending upper T range.")
    if warn_tmin:
        print("[WARN] Tmin slice contains liquid; consider extending lower T range.")


def main_viz() -> None:
    system_name = "Al-Hf-Nb-W-Zr"
    element_names = ["Al", "Hf", "Nb", "W", "Zr"]
    grid_delta = float(os.getenv("HEA_GRID_DELTA", "0.05"))
    tol = float(os.getenv("HEA_SLICE_TOL", str(max(0.5 * grid_delta, 1e-6))))
    use_phase_extrema = os.getenv("HEA_SLICE_PHASE_EXTREMA", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

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

    html_out_dir = plotter_dir / "hea_dump"
    html_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HEA MAIN_VIZ")
    print("=" * 80)
    print(f"Phase boundary file: {phase_boundary_file}")
    print(f"HTML output dir: {html_out_dir}")
    print(f"Grid delta: {grid_delta}")
    print(f"Slice tolerance: {tol}")
    print(f"Slice phase-extrema filter: {use_phase_extrema}")

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

    # ------------------ Binary batches (10 binaries x 10 variants) ------------------
    binary_variant_targets = [
        (0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0),
        (0.0, 0.05, 0.0),
        (0.0, 0.0, 0.05),
        (0.05, 0.05, 0.0),
        (0.05, 0.0, 0.05),
        (0.0, 0.05, 0.05),
        (0.10, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, 0.0, 0.10),
    ]

    bin_saved = 0
    for a, b in combinations(full_names, 2):
        others = [x for x in full_names if x not in {a, b}]
        for vidx, target in enumerate(binary_variant_targets, start=1):
            fixed = {}
            for name, val in zip(others, target):
                snapped = np.round(val / grid_delta) * grid_delta
                fixed[name] = _nearest_available(full_df[name], float(snapped))
            try:
                fig = plotter.plot_binary_slice_tx(
                    comp_a=a,
                    comp_b=b,
                    fixed_components=fixed,
                    tolerance=tol,
                    phase_extrema_filter=use_phase_extrema,
                    title=f"{a}-{b} binary | fixed {fixed}",
                )
            except Exception as e:
                print(f"[SKIP][binary] {a}-{b} variant {vidx}: {e}")
                continue
            out = html_out_dir / f"binary_{a}_{b}_v{vidx:02d}.html"
            fig.write_html(str(out), include_plotlyjs="cdn")
            bin_saved += 1

    # ------------------ Ternary batches (10 ternaries x 5 variants) ------------------
    ternary_variant_targets = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.0, 0.05),
        (0.10, 0.0),
        (0.0, 0.10),
    ]

    tern_saved = 0
    for a, b, c in combinations(full_names, 3):
        others = [x for x in full_names if x not in {a, b, c}]
        for vidx, target in enumerate(ternary_variant_targets, start=1):
            fixed = {}
            for name, val in zip(others, target):
                snapped = np.round(val / grid_delta) * grid_delta
                fixed[name] = _nearest_available(full_df[name], float(snapped))
            try:
                fig = plotter.plot_ternary_slice(
                    comp_a=a,
                    comp_b=b,
                    comp_c=c,
                    fixed_components=fixed,
                    tolerance=tol,
                    phase_extrema_filter=use_phase_extrema,
                    title=f"{a}-{b}-{c} ternary | fixed {fixed}",
                    color_by="Phase",
                )
            except Exception as e:
                print(f"[SKIP][ternary] {a}-{b}-{c} variant {vidx}: {e}")
                continue
            out = html_out_dir / f"ternary_{a}_{b}_{c}_v{vidx:02d}.html"
            fig.write_html(str(out), include_plotlyjs="cdn")
            tern_saved += 1

    # ------------------ Quaternary batches (5 quaternaries x 3 variants) ------------
    quaternary_fixed_targets = [0.0, 0.05, 0.10]
    quat_saved = 0

    for selected4 in combinations(full_names, 4):
        omitted = [x for x in full_names if x not in set(selected4)]
        if len(omitted) != 1:
            continue
        fixed_name = omitted[0]

        for vidx, target in enumerate(quaternary_fixed_targets, start=1):
            snapped = np.round(target / grid_delta) * grid_delta
            fixed_val = _nearest_available(full_df[fixed_name], float(snapped))
            fixed = {fixed_name: fixed_val}

            try:
                df_slice = plotter.filter_by_fixed_components(fixed_components=fixed, tolerance=tol)
                df_q = _reduce_to_quaternary_df(df_slice, selected_full_components=list(selected4))
                if df_q.empty:
                    raise ValueError("Projected quaternary slice is empty")
                qplotter = PhaseBoundaryPlotter(
                    equilibrium_df=df_q,
                    composition_cols=["x0", "x1", "x2"],
                    element_names=list(selected4),
                )
                fig_l = qplotter.plot_quaternary_phase_tetrahedral(
                    phase_filter="L",
                    temperature_extrema="min",
                    composition_tol=tol,
                    title=f"Quaternary {selected4} | {fixed_name}={fixed_val:.2f} | Liquid Tmin",
                )
                fig_b = qplotter.plot_quaternary_phase_tetrahedral(
                    phase_filter="BCC*",
                    temperature_extrema="max",
                    composition_tol=tol,
                    title=f"Quaternary {selected4} | {fixed_name}={fixed_val:.2f} | BCC Tmax",
                )
            except Exception as e:
                print(f"[SKIP][quaternary] {selected4} variant {vidx}: {e}")
                continue

            out_l = html_out_dir / f"quaternary_{'_'.join(selected4)}_v{vidx:02d}_liquid_min.html"
            out_b = html_out_dir / f"quaternary_{'_'.join(selected4)}_v{vidx:02d}_bcc_max.html"
            fig_l.write_html(str(out_l), include_plotlyjs="cdn")
            fig_b.write_html(str(out_b), include_plotlyjs="cdn")
            quat_saved += 2

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
    mode = os.getenv("HEA_MAIN", "viz").strip().lower()
    if mode == "post":
        main_post()
    elif mode == "viz":
        main_viz()
    else:
        raise ValueError(f"Invalid HEA_MAIN value: {mode}. Use 'post' or 'viz'.")
