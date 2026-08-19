from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from gliq_mapp_compare import resolve_gliq_input


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALC_DIR = REPO_ROOT / "all_dumps" / "gliq_manu_forreal_plusML_correction"
MAPP_REPOSITORY = "https://github.com/qjhong/mapp_api.git"


def export_formulas(gliq_results: Path, formula_file: Path) -> int:
    results = pd.read_excel(gliq_results)
    if "reduced_formula" not in results.columns:
        raise KeyError(f"Missing reduced_formula column in {gliq_results}")
    formulas = results["reduced_formula"].dropna().astype(str)
    if formulas.duplicated().any():
        raise ValueError("MAPP workflow requires unique reduced_formula values.")
    if len(formulas) > 10_000:
        raise ValueError("MAPP accepts at most 10,000 formulas per run.")

    formula_file.parent.mkdir(parents=True, exist_ok=True)
    formulas.to_csv(formula_file, index=False, header=False)
    return len(formulas)


def ensure_mapp_script(mapp_repo: Path) -> Path:
    script = mapp_repo / "mapp_mp.py"
    if not script.exists():
        mapp_repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", MAPP_REPOSITORY, str(mapp_repo)], check=True)
    if not script.exists():
        raise FileNotFoundError(f"Official MAPP script not found at {script}")
    return script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official MAPP predictor and compare it with corrected or regular GLIQ results.")
    parser.add_argument("--calc-dir", type=Path, default=DEFAULT_CALC_DIR)
    parser.add_argument("--gliq-results", type=Path)
    parser.add_argument("--gliq-column")
    parser.add_argument("--mapp-repo", type=Path)
    parser.add_argument("--mapp-python", default=sys.executable, help="Python interpreter containing pandas and requests.")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true", help="Reuse an existing mapp_run/output.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calc_dir = args.calc_dir.resolve()
    gliq_results, gliq_column = resolve_gliq_input(calc_dir, args.gliq_results, args.gliq_column)
    mapp_repo = (args.mapp_repo or calc_dir / "mapp_api").resolve()
    run_dir = calc_dir / "mapp_run"
    formula_file = run_dir / "chemical_formula.csv"
    output_file = run_dir / "output.csv"

    formula_count = export_formulas(gliq_results, formula_file)
    print(f"Exported {formula_count} formulas to {formula_file}")
    if args.prepare_only:
        return

    if not args.skip_prediction:
        mapp_script = ensure_mapp_script(mapp_repo)
        subprocess.run([args.mapp_python, str(mapp_script)], cwd=run_dir, check=True)
    if not output_file.exists():
        raise FileNotFoundError(f"MAPP prediction output not found at {output_file}")

    compare_script = Path(__file__).with_name("gliq_mapp_compare.py")
    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--calc-dir",
            str(calc_dir),
            "--gliq-results",
            str(gliq_results),
            "--gliq-column",
            gliq_column,
            "--mapp-output",
            str(output_file),
            "--figures-dir",
            str(args.figures_dir.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
