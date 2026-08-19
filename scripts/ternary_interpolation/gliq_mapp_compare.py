from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALC_DIR = REPO_ROOT / "all_dumps" / "gliq_manu_forreal_plusML"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def resolve_gliq_input(calc_dir: Path, gliq_results: Path | None, gliq_column: str | None) -> tuple[Path, str]:
    if gliq_results is None:
        candidates = [
            (calc_dir / "optimized_l0_tern_results.xlsx", "final_gliq_temp"),
            (calc_dir / "ternary_Gliq_mps_final_linear.xlsx", "gliq_melting_temp"),
        ]
        try:
            gliq_results, default_column = next(candidate for candidate in candidates if candidate[0].exists())
        except StopIteration as exc:
            raise FileNotFoundError(f"No corrected or regular GLIQ results workbook found in {calc_dir}") from exc
        gliq_column = gliq_column or default_column
    else:
        gliq_results = gliq_results.resolve()

    if gliq_column is None:
        columns = pd.read_excel(gliq_results, nrows=0).columns
        try:
            gliq_column = next(column for column in ["final_gliq_temp", "gliq_melting_temp"] if column in columns)
        except StopIteration as exc:
            raise KeyError(f"Could not identify a GLIQ prediction column in {gliq_results}") from exc
    return gliq_results.resolve(), gliq_column


def compute_metrics(df: pd.DataFrame, pred_col: str, exp_col: str) -> dict:
    valid = df[[pred_col, exp_col]].dropna()
    if valid.empty:
        raise ValueError(f"No valid rows available for {pred_col} vs {exp_col}.")

    errors = valid[pred_col] - valid[exp_col]
    abs_errors = errors.abs()
    nonzero_exp = valid[exp_col] != 0
    denom = np.sum((valid[exp_col] - valid[exp_col].mean()) ** 2)
    return {
        "n": len(valid),
        "rmse_k": np.sqrt(np.mean(errors**2)),
        "mae_k": abs_errors.mean(),
        "mape_pct": np.mean(abs_errors[nonzero_exp] / valid.loc[nonzero_exp, exp_col].abs()) * 100,
        "median_abs_error_k": abs_errors.median(),
        "bias_k": errors.mean(),
        "error_std_k": errors.std(ddof=1),
        "r2": 1.0 - np.sum(errors**2) / denom if denom > 0 else np.nan,
    }


def add_row_errors(df: pd.DataFrame, pred_col: str, prefix: str) -> None:
    df[f"{prefix}_error_k"] = df[pred_col] - df["melting_point_k"]
    df[f"{prefix}_abs_error_k"] = df[f"{prefix}_error_k"].abs()
    df[f"{prefix}_abs_pct_error"] = np.where(
        df["melting_point_k"] != 0,
        df[f"{prefix}_abs_error_k"] / df["melting_point_k"].abs() * 100,
        np.nan,
    )


def load_comparison(gliq_results: Path, mapp_output: Path, gliq_column: str) -> pd.DataFrame:
    gliq_df = pd.read_excel(gliq_results)
    required_gliq = {"reduced_formula", "melting_point_k", gliq_column}
    missing_gliq = required_gliq - set(gliq_df.columns)
    if missing_gliq:
        raise KeyError(f"Missing GLIQ columns in {gliq_results}: {sorted(missing_gliq)}")
    if gliq_df["reduced_formula"].duplicated().any():
        raise ValueError("GLIQ formulas must be unique for a one-to-one MAPP comparison.")

    mapp_df = pd.read_csv(mapp_output).drop(columns=["Unnamed: 0"], errors="ignore")
    required_mapp = {"chemical_formula", "melting_temperature_in_kelvin"}
    missing_mapp = required_mapp - set(mapp_df.columns)
    if missing_mapp:
        raise KeyError(f"Missing MAPP columns in {mapp_output}: {sorted(missing_mapp)}")
    if mapp_df["chemical_formula"].duplicated().any():
        raise ValueError("MAPP formulas must be unique for a one-to-one comparison.")

    mapp_df = mapp_df.rename(
        columns={
            "chemical_formula": "reduced_formula",
            "melting_temperature_in_kelvin": "mapp_melting_temp",
            "standard_error_in_kelvin": "mapp_standard_error_k",
        }
    )
    comparison = gliq_df.merge(mapp_df, on="reduced_formula", how="left", validate="one_to_one")
    missing_predictions = comparison.loc[comparison["mapp_melting_temp"].isna(), "reduced_formula"].tolist()
    if missing_predictions:
        raise ValueError(f"MAPP output is missing {len(missing_predictions)} formulas: {missing_predictions[:10]}")

    comparison["gliq_melting_temp"] = comparison[gliq_column]
    add_row_errors(comparison, "gliq_melting_temp", "gliq")
    add_row_errors(comparison, "mapp_melting_temp", "mapp")
    return comparison


def make_comparison_figure(comparison: pd.DataFrame, figures_dir: Path, output_name: str) -> None:
    plot_columns = ["melting_point_k", "gliq_melting_temp", "mapp_melting_temp"]
    valid = comparison[plot_columns].dropna()
    lim_min = valid.min().min()
    lim_max = valid.max().max()
    pad = 0.03 * (lim_max - lim_min) if lim_max > lim_min else 10.0
    limits = (lim_min - pad, lim_max + pad)

    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, column, title in zip(
        axes,
        ["gliq_melting_temp", "mapp_melting_temp"],
        ["G-Liquid", "MAPP"],
    ):
        ax.scatter(
            valid["melting_point_k"],
            valid[column],
            c="tab:blue",
            s=80,
            edgecolors="none",
            linewidths=0,
        )
        ax.plot(limits, limits, "k--", linewidth=2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=15, width=1.0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("Predicted Melting Temperature (K)", fontsize=15, fontweight="bold")
    fig.supxlabel("MPDS Congruent Melting Temperature (K)", fontsize=15, fontweight="bold")
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_base = figures_dir / output_name
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare corrected or regular GLIQ and MAPP melting temperatures against MPDS.")
    parser.add_argument("--calc-dir", type=Path, default=DEFAULT_CALC_DIR)
    parser.add_argument("--gliq-results", type=Path)
    parser.add_argument("--mapp-output", type=Path)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--gliq-column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calc_dir = args.calc_dir.resolve()
    gliq_results, gliq_column = resolve_gliq_input(calc_dir, args.gliq_results, args.gliq_column)
    mapp_output = (args.mapp_output or calc_dir / "mapp_run" / "output.csv").resolve()
    comparison = load_comparison(gliq_results, mapp_output, gliq_column)

    summary = pd.DataFrame(
        [
            {"model": "GLIQ", **compute_metrics(comparison, "gliq_melting_temp", "melting_point_k")},
            {"model": "MAPP", **compute_metrics(comparison, "mapp_melting_temp", "melting_point_k")},
        ]
    )
    calc_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(calc_dir / "mapp_gliq_comparison.csv", index=False)
    summary.to_csv(calc_dir / "mapp_gliq_metrics_summary.csv", index=False)
    figure_name = f"{calc_dir.name}_mapp_mpds_comparison"
    make_comparison_figure(comparison, args.figures_dir.resolve(), figure_name)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nDetailed comparison: {calc_dir / 'mapp_gliq_comparison.csv'}")
    print(f"Metrics summary: {calc_dir / 'mapp_gliq_metrics_summary.csv'}")
    print(f"Figures: {args.figures_dir.resolve() / figure_name}.png and .svg")


if __name__ == "__main__":
    main()
