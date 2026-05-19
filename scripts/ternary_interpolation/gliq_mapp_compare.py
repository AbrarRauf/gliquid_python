from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


calc_path = "all_dumps/gliq_manu_forreal/"
inter_doc = calc_path + "ternary_Gliq_mps_final_linear.xlsx"


def compute_metrics(df: pd.DataFrame, pred_col: str, exp_col: str) -> dict:
	"""Compute error statistics for a predictor column against experiment."""
	valid = df[[pred_col, exp_col]].dropna()
	if valid.empty:
		raise ValueError(f"No valid rows available for {pred_col} vs {exp_col}.")

	errors = valid[pred_col] - valid[exp_col]
	abs_errors = errors.abs()
	sq_errors = errors**2
	denom = np.sum((valid[exp_col] - valid[exp_col].mean()) ** 2)
	rmse = float(np.sqrt(np.mean(errors**2)))
	mae = float(np.mean(abs_errors))
	medae = float(np.median(abs_errors))
	bias = float(np.mean(errors))
	error_std = float(errors.std(ddof=1))
	pred_std = float(valid[pred_col].std(ddof=1))
	exp_std = float(valid[exp_col].std(ddof=1))
	r2 = float(1.0 - (np.sum(sq_errors) / denom)) if denom > 0 else float("nan")
	nrmse_mean = float(rmse / valid[exp_col].mean()) if valid[exp_col].mean() != 0 else float("nan")

	return {
		"n": int(valid.shape[0]),
		"rmse": rmse,
		"mae": mae,
		"medae": medae,
		"bias": bias,
		"error_std": error_std,
		"pred_std": pred_std,
		"exp_std": exp_std,
		"r2": r2,
		"nrmse_mean": nrmse_mean,
	}


def add_row_error_metrics(df: pd.DataFrame, pred_col: str, exp_col: str, prefix: str) -> None:
	"""Add per-row error metrics to the dataframe for one prediction method."""
	error_col = f"{prefix}_error"
	abs_error_col = f"{prefix}_abs_error"
	sq_error_col = f"{prefix}_sq_error"
	ape_col = f"{prefix}_abs_pct_error"

	df[error_col] = df[pred_col] - df[exp_col]
	df[abs_error_col] = df[error_col].abs()
	df[sq_error_col] = df[error_col] ** 2

	# Avoid divide-by-zero for percentage error.
	df[ape_col] = np.where(df[exp_col] != 0, (df[abs_error_col] / df[exp_col]) * 100.0, np.nan)


def make_scatter_plot(
	df: pd.DataFrame,
	x_col: str,
	y_col: str,
	x_label: str,
	y_label: str,
	out_base: Path,
	title: str = "",
) -> None:
	"""Create and save a manuscript-ready scatter plot with y=x reference."""
	valid = df[[x_col, y_col]].dropna()
	if valid.empty:
		raise ValueError(f"No valid rows available for {y_col} vs {x_col}.")

	x = valid[x_col].to_numpy()
	y = valid[y_col].to_numpy()
	lim_min = float(min(np.min(x), np.min(y)))
	lim_max = float(max(np.max(x), np.max(y)))
	pad = 0.03 * (lim_max - lim_min) if lim_max > lim_min else 10.0

	plt.rcParams.update(
		{
			"font.size": 16,
			"axes.labelsize": 18,
			"axes.titlesize": 18,
			"xtick.labelsize": 15,
			"ytick.labelsize": 15,
		}
	)

	fig, ax = plt.subplots(figsize=(7.5, 7.0))
	ax.scatter(x, y, s=70, alpha=0.9, edgecolor="black", linewidth=0.7, color="#1f77b4")
	ax.plot(
		[lim_min - pad, lim_max + pad],
		[lim_min - pad, lim_max + pad],
		linestyle="--",
		color="black",
		linewidth=2.0,
		label="y=x",
	)

	ax.set_xlim(lim_min - pad, lim_max + pad)
	ax.set_ylim(lim_min - pad, lim_max + pad)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_title(title)
	ax.tick_params(axis="both", which="major", width=1.6, length=6)

	for spine in ax.spines.values():
		spine.set_linewidth(1.8)

	ax.legend(frameon=False, fontsize=13, loc="upper left")
	fig.tight_layout()

	fig.savefig(out_base.with_suffix(".png"), dpi=600)
	fig.savefig(out_base.with_suffix(".svg"))
	plt.close(fig)


def main() -> None:
	out_dir = Path(calc_path)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_excel(inter_doc)

	required_cols = ["melting_point_k", "gliq_melting_temp", "mapp_melting_temp"]
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		raise KeyError(f"Missing required columns in {inter_doc}: {missing_cols}")

	add_row_error_metrics(df, pred_col="gliq_melting_temp", exp_col="melting_point_k", prefix="gliq")
	add_row_error_metrics(df, pred_col="mapp_melting_temp", exp_col="melting_point_k", prefix="mapp")

	gliq_metrics = compute_metrics(df, pred_col="gliq_melting_temp", exp_col="melting_point_k")
	mapp_metrics = compute_metrics(df, pred_col="mapp_melting_temp", exp_col="melting_point_k")

	summary_df = pd.DataFrame(
		[
			{"model": "GLIQ", **gliq_metrics},
			{"model": "MAPP", **mapp_metrics},
		]
	)
	summary_df.to_csv(out_dir / "melting_point_metrics_summary.csv", index=False)
	df.to_excel(out_dir / "ternary_Gliq_mps_final_linear_with_metrics.xlsx", index=False)

	make_scatter_plot(
		df,
		x_col="melting_point_k",
		y_col="gliq_melting_temp",
		x_label="Experimental melting point (K)",
		y_label="GLIQ melting temperature (K)",
		out_base=out_dir / "gliq_vs_exp_scatter",
	)
	make_scatter_plot(
		df,
		x_col="melting_point_k",
		y_col="mapp_melting_temp",
		x_label="Experimental melting point (K)",
		y_label="MAPP melting temperature (K)",
		out_base=out_dir / "mapp_vs_exp_scatter",
	)

	print("=== GLIQ vs Experimental ===")
	print(f"N = {gliq_metrics['n']}")
	print(f"RMSE (K): {gliq_metrics['rmse']:.3f}")
	print(f"MAE (K): {gliq_metrics['mae']:.3f}")
	print(f"MedAE (K): {gliq_metrics['medae']:.3f}")
	print(f"Bias / Mean error (K): {gliq_metrics['bias']:.3f}")
	print(f"Std of errors (K): {gliq_metrics['error_std']:.3f}")
	print(f"Std of GLIQ predictions (K): {gliq_metrics['pred_std']:.3f}")
	print(f"Std of experimental values (K): {gliq_metrics['exp_std']:.3f}")
	print(f"R^2: {gliq_metrics['r2']:.3f}")
	print(f"NRMSE (RMSE/mean experimental): {gliq_metrics['nrmse_mean']:.3f}")
	print()

	print("=== MAPP vs Experimental ===")
	print(f"N = {mapp_metrics['n']}")
	print(f"RMSE (K): {mapp_metrics['rmse']:.3f}")
	print(f"MAE (K): {mapp_metrics['mae']:.3f}")
	print(f"MedAE (K): {mapp_metrics['medae']:.3f}")
	print(f"Bias / Mean error (K): {mapp_metrics['bias']:.3f}")
	print(f"Std of errors (K): {mapp_metrics['error_std']:.3f}")
	print(f"Std of MAPP predictions (K): {mapp_metrics['pred_std']:.3f}")
	print(f"Std of experimental values (K): {mapp_metrics['exp_std']:.3f}")
	print(f"R^2: {mapp_metrics['r2']:.3f}")
	print(f"NRMSE (RMSE/mean experimental): {mapp_metrics['nrmse_mean']:.3f}")


if __name__ == "__main__":
	main()

