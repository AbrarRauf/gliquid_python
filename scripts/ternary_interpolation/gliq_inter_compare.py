import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

linear_df = pd.read_excel("all_dumps/gliq_manu_forreal/ternary_Gliq_mps_final_linear.xlsx")
muggianu_df = pd.read_excel("all_dumps/gliq_manu_forreal_muggianu/ternary_Gliq_mps_final_muggianu.xlsx")
kohler_df = pd.read_excel("all_dumps/gliq_manu_forreal_kohler/ternary_Gliq_mps_final_kohler.xlsx")

plot_dump_dir = Path("all_dumps/gliq_inter_compare")


def compute_error_metrics(df: pd.DataFrame, pred_col: str = "gliq_melting_temp", ref_col: str = "melting_point_k") -> dict:
	required_cols = {pred_col, ref_col}
	missing = required_cols - set(df.columns)
	if missing:
		raise KeyError(f"Missing required columns: {sorted(missing)}")

	valid = df[[pred_col, ref_col]].dropna()
	if valid.empty:
		raise ValueError("No valid rows after dropping NaN values.")

	errors = valid[pred_col] - valid[ref_col]
	mae = float(np.mean(np.abs(errors)))
	bias = float(np.mean(errors))
	error_std = float(errors.std(ddof=1))

	return {
		"n": int(valid.shape[0]),
		"mae": mae,
		"bias": bias,
		"error_std": error_std,
	}


def make_scatter_plot(
	df: pd.DataFrame,
	approach_name: str,
	x_col: str = "melting_point_k",
	y_col: str = "gliq_melting_temp",
) -> None:
	valid = df[[x_col, y_col]].dropna()
	if valid.empty:
		raise ValueError(f"No valid rows for plotting {approach_name}.")

	x = valid[x_col].to_numpy()
	y = valid[y_col].to_numpy()
	lim_min = float(min(np.min(x), np.min(y)))
	lim_max = float(max(np.max(x), np.max(y)))
	pad = 0.03 * (lim_max - lim_min) if lim_max > lim_min else 10.0

	fig, ax = plt.subplots(figsize=(7.0, 6.5))
	ax.scatter(x, y, s=70, alpha=0.9, edgecolor="black", linewidth=0.7, color="#1f77b4")
	ax.plot(
		[lim_min - pad, lim_max + pad],
		[lim_min - pad, lim_max + pad],
		linestyle="--",
		color="black",
		linewidth=1.8,
		label="y=x",
	)

	ax.set_xlim(lim_min - pad, lim_max + pad)
	ax.set_ylim(lim_min - pad, lim_max + pad)
	ax.set_xlabel("melting_point_k (K)")
	ax.set_ylabel("gliq_melting_temp (K)")
	ax.set_title(f"{approach_name.capitalize()} interpolation")
	ax.legend(frameon=False)

	for spine in ax.spines.values():
		spine.set_linewidth(1.3)

	fig.tight_layout()
	fig.savefig(plot_dump_dir / f"gliq_vs_exp_scatter_{approach_name}.png", dpi=600)
	fig.savefig(plot_dump_dir / f"gliq_vs_exp_scatter_{approach_name}.svg")
	plt.close(fig)


def main() -> None:
	plot_dump_dir.mkdir(parents=True, exist_ok=True)

	approaches = {
		"linear": linear_df,
		"muggianu": muggianu_df,
		"kohler": kohler_df,
	}

	rows = []
	for name, df in approaches.items():
		metrics = compute_error_metrics(df)
		rows.append({"approach": name, **metrics})
		make_scatter_plot(df=df, approach_name=name)

	comparison_df = pd.DataFrame(rows)

	print("=== GLIQ interpolation approach comparison ===")
	print("Reference: melting_point_k | Prediction: gliq_melting_temp")
	print()
	print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	print()
	print(f"Scatter plots saved to: {plot_dump_dir}")


if __name__ == "__main__":
	main()

