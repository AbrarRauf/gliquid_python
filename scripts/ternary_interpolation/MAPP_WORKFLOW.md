# GLIQ–MAPP comparison workflow

Run from the `gliquid_python` repository root. The default input is
`all_dumps/gliq_manu_forreal_plusML_correction/optimized_l0_tern_results.xlsx`.
Corrected directories are detected from `optimized_l0_tern_results.xlsx` and use
`final_gliq_temp`. Regular directories are detected from `ternary_Gliq_mps_final_linear.xlsx` and
use `gliq_melting_temp`.

## Complete run

```powershell
python scripts/ternary_interpolation/mapp_workflow.py
```

On the first run, the workflow clones the official
[`qjhong/mapp_api`](https://github.com/qjhong/mapp_api) repository into the ignored calculation
directory. It exports `mapp_run/chemical_formula.csv`, runs the official `mapp_mp.py`, and then runs
the independent comparison.

To use a separate MAPP environment:

```powershell
python scripts/ternary_interpolation/mapp_workflow.py --mapp-python C:\path\to\mapp-env\python.exe
```

To run the regular, uncorrected calculation:

```powershell
python scripts/ternary_interpolation/mapp_workflow.py --calc-dir all_dumps/gliq_manu_forreal_plusML
```

To export formulas without contacting MAPP:

```powershell
python scripts/ternary_interpolation/mapp_workflow.py --prepare-only
```

To reuse an existing `mapp_run/output.csv` and regenerate comparisons quickly:

```powershell
python scripts/ternary_interpolation/mapp_workflow.py --skip-prediction
```

## Outputs

The calculation directory receives:

- `mapp_run/chemical_formula.csv`
- `mapp_run/output.csv`
- `mapp_gliq_comparison.csv`
- `mapp_gliq_metrics_summary.csv`

The repository `figures/` directory receives 300-DPI PNG and SVG files prefixed with the calculation
directory name, so corrected and regular comparisons do not overwrite each other.
