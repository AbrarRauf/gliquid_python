"""Post-processing helpers for stitched General HSX lower-hull cache artifacts."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _infer_comp_cols(df: pd.DataFrame) -> List[str]:
	cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
	return sorted(cols, key=lambda c: int(c[1:]))


def _phase_is_liquid(phase: Any, liquid_aliases: Sequence[str]) -> bool:
	return str(phase).strip().upper() in {s.strip().upper() for s in liquid_aliases}


def _round_sig(value: float, sig: int = 3) -> float:
	v = float(value)
	if not np.isfinite(v) or v == 0.0:
		return v
	return float(f"{v:.{sig}g}")


def _infer_grid_delta(df: pd.DataFrame, comp_cols: Sequence[str]) -> float:
	deltas = []
	for col in comp_cols:
		vals = np.sort(df[col].dropna().unique().astype(float))
		if len(vals) < 2:
			continue
		dv = np.diff(vals)
		dv = dv[dv > 1e-12]
		if len(dv):
			deltas.append(float(np.min(dv)))
	return min(deltas) if deltas else 0.0


def _connected_components(points: np.ndarray, threshold: float) -> List[List[int]]:
	if len(points) == 0:
		return []
	threshold = max(float(threshold), 0.0)
	if threshold == 0.0:
		return [[i] for i in range(len(points))]

	tree = cKDTree(points)
	neighbors = tree.query_ball_point(points, r=threshold + 1e-12)
	seen = np.zeros(len(points), dtype=bool)

	components: List[List[int]] = []
	for i in range(len(points)):
		if seen[i]:
			continue
		stack = [i]
		seen[i] = True
		comp = []
		while stack:
			cur = stack.pop()
			comp.append(cur)
			for n in neighbors[cur]:
				if not seen[n]:
					seen[n] = True
					stack.append(n)
		components.append(sorted(comp))

	return components


@dataclass
class CacheBlock:
	stem: str
	manifest_path: Path
	equilibrium_path: Path
	simplex_path: Path
	gtx_path: Optional[Path]
	t_min: float
	t_max: float
	n_gtx_rows: int
	n_equilibrium_rows: int
	n_simplex_rows: int
	manifest: Dict[str, Any]


class GeneralHSXPostprocess:
	"""Load and stitch lower-hull cache artifacts for one system.

	Assumed input layout:
	- a system directory containing one full-range cache set directly, or
	- a system directory containing subdirectories with per-temperature-block sets.
	"""

	def __init__(
		self,
		system_cache_dir: str,
		recursive: bool = True,
		load_gtx: bool = True,
		liquid_aliases: Optional[Sequence[str]] = None,
	):
		self.system_cache_dir = Path(system_cache_dir)
		self.recursive = bool(recursive)
		self.load_gtx = bool(load_gtx)
		self.liquid_aliases = tuple(liquid_aliases or ["L", "LIQUID"])

		self.blocks: List[CacheBlock] = []
		self.block_index_df: pd.DataFrame = pd.DataFrame()

		self.equilibrium_df: pd.DataFrame = pd.DataFrame()
		self.simplex_df: pd.DataFrame = pd.DataFrame()
		self.gtx_df: pd.DataFrame = pd.DataFrame()

		self.composition_cols: List[str] = []
		self.system_name: Optional[str] = None

		self.global_eutectic: Dict[str, Any] = {}
		self.low_t_clusters: Dict[str, Any] = {}
		self.phase_boundary_equilibrium_df: pd.DataFrame = pd.DataFrame()
		self.phase_boundary_equilibrium_path: Optional[Path] = None
		self.tmax_filter_diagnostic: Dict[str, Any] = {}

	@classmethod
	def from_system_cache_dir(
		cls,
		system_cache_dir: str,
		recursive: bool = True,
		load_gtx: bool = True,
		liquid_aliases: Optional[Sequence[str]] = None,
	) -> "GeneralHSXPostprocess":
		obj = cls(
			system_cache_dir=system_cache_dir,
			recursive=recursive,
			load_gtx=load_gtx,
			liquid_aliases=liquid_aliases,
		)
		obj.load_and_stitch()
		return obj

	def _discover_manifest_paths(self) -> List[Path]:
		if not self.system_cache_dir.exists() or not self.system_cache_dir.is_dir():
			raise FileNotFoundError(f"System cache directory not found: {self.system_cache_dir}")

		if self.recursive:
			manifests = sorted(self.system_cache_dir.rglob("*_manifest.json"))
		else:
			manifests = sorted(self.system_cache_dir.glob("*_manifest.json"))

		if not manifests:
			raise FileNotFoundError(
				f"No *_manifest.json files found in {self.system_cache_dir} (recursive={self.recursive})."
			)
		return manifests

	@staticmethod
	def _stem_from_manifest(manifest_path: Path) -> str:
		name = manifest_path.name
		suffix = "_manifest.json"
		if not name.endswith(suffix):
			raise ValueError(f"Unexpected manifest filename: {manifest_path}")
		return name[: -len(suffix)]

	def _build_block(self, manifest_path: Path) -> CacheBlock:
		stem = self._stem_from_manifest(manifest_path)
		parent = manifest_path.parent

		equilibrium_path = parent / f"{stem}_equilibrium.pkl.gz"
		simplex_path = parent / f"{stem}_simplex.pkl.gz"
		gtx_path = parent / f"{stem}_gtx.pkl.gz"

		missing = [
			str(p)
			for p in [equilibrium_path, simplex_path]
			if not p.exists()
		]
		if missing:
			raise FileNotFoundError(
				f"Missing required cache files for stem {stem}: {missing}"
			)

		with open(manifest_path, "r", encoding="utf-8") as f:
			manifest = json.load(f)

		temp_range = manifest.get("temp_range")
		if not temp_range or len(temp_range) != 2:
			raise ValueError(f"Manifest missing temp_range for stem {stem}: {manifest_path}")

		if self.load_gtx and not gtx_path.exists():
			raise FileNotFoundError(
				f"GTX file required for stitching source_row_id but not found: {gtx_path}"
			)

		return CacheBlock(
			stem=stem,
			manifest_path=manifest_path,
			equilibrium_path=equilibrium_path,
			simplex_path=simplex_path,
			gtx_path=gtx_path if gtx_path.exists() else None,
			t_min=float(temp_range[0]),
			t_max=float(temp_range[1]),
			n_gtx_rows=int(manifest.get("n_gtx_rows", 0)),
			n_equilibrium_rows=int(manifest.get("n_equilibrium_rows", 0)),
			n_simplex_rows=int(manifest.get("n_simplex_rows", 0)),
			manifest=manifest,
		)

	@staticmethod
	def _validate_temp_ranges(blocks: Sequence[CacheBlock]):
		if not blocks:
			return
		for i in range(1, len(blocks)):
			prev = blocks[i - 1]
			cur = blocks[i]
			if cur.t_min < prev.t_max - 1e-9:
				raise ValueError(
					"Overlapping temperature blocks detected: "
					f"{prev.stem} [{prev.t_min}, {prev.t_max}] vs "
					f"{cur.stem} [{cur.t_min}, {cur.t_max}]"
				)
			if cur.t_min > prev.t_max + 1e-9:
				warnings.warn(
					"Gap between temperature blocks: "
					f"{prev.stem} -> {cur.stem} ({prev.t_max} to {cur.t_min})"
				)

	@staticmethod
	def _offset_vertex_source_ids(values: Any, offset: int) -> Any:
		if offset == 0:
			return values
		if isinstance(values, (list, tuple, np.ndarray)):
			return [int(v) + int(offset) for v in values]
		return values

	def _compute_global_eutectic(self, interior_tol: float = 1e-6) -> Dict[str, Any]:
		if self.equilibrium_df.empty or self.simplex_df.empty:
			return {
				"eutectic_temperature": None,
				"eutectic_temperature_C": None,
				"eutectic_composition_independent": None,
				"eutectic_composition_full": None,
				"simplex_ids": [],
				"coexisting_solids": [],
				"n_liquid_points_at_temperature": 0,
			}

		simplex_scan = self.simplex_df.sort_values(by=["T_K", "simplex_id"], ascending=True)
		for _, simplex_meta in simplex_scan.iterrows():
			simplex_id = int(simplex_meta["simplex_id"])
			simplex_rows = self.equilibrium_df[self.equilibrium_df["simplex_id"] == simplex_id].copy()
			if simplex_rows.empty:
				continue

			liq_t = simplex_rows[
				simplex_rows["Phase"].apply(lambda p: _phase_is_liquid(p, self.liquid_aliases))
			].copy()
			if liq_t.empty:
				continue

			liq_t = (
				liq_t
				.sort_values(by=["G"] + self.composition_cols, ascending=True)
				.drop_duplicates(subset=self.composition_cols, keep="first")
				.reset_index(drop=True)
			)

			indep_mean = liq_t[self.composition_cols].mean(axis=0)
			indep_vals_raw = [float(indep_mean[col]) for col in self.composition_cols]
			dep_val_raw = float(1.0 - np.sum(indep_vals_raw))

			if not all(v > interior_tol for v in [dep_val_raw] + indep_vals_raw):
				continue

			indep_vals = [_round_sig(v) for v in indep_vals_raw]
			dep_val = _round_sig(dep_val_raw)
			full_comp = [dep_val] + indep_vals

			coexisting_solids = sorted({
				p for p in simplex_rows["Phase"].astype(str).unique().tolist()
				if not _phase_is_liquid(p, self.liquid_aliases)
			})

			temp_val = float(simplex_rows["T_K"].iloc[0])
			return {
				"eutectic_temperature": float(temp_val),
				"eutectic_temperature_C": float(temp_val - 273.15),
				"eutectic_composition_independent": indep_vals,
				"eutectic_composition_full": full_comp,
				"simplex_ids": [simplex_id],
				"coexisting_solids": coexisting_solids,
				"n_liquid_points_at_temperature": int(len(liq_t)),
			}

		return {
			"eutectic_temperature": None,
			"eutectic_temperature_C": None,
			"eutectic_composition_independent": None,
			"eutectic_composition_full": None,
			"simplex_ids": [],
			"coexisting_solids": [],
			"n_liquid_points_at_temperature": 0,
		}

	def _compute_low_t_clusters(self) -> Dict[str, Any]:
		if self.equilibrium_df.empty:
			return {
				"tmin": None,
				"cluster_records": pd.DataFrame(),
				"coexisting_solids_union": [],
			}

		liq_df = self.equilibrium_df[
			self.equilibrium_df["Phase"].apply(lambda p: _phase_is_liquid(p, self.liquid_aliases))
		].copy()
		if liq_df.empty:
			return {
				"tmin": None,
				"cluster_records": pd.DataFrame(),
				"coexisting_solids_union": [],
			}

		tmin = float(liq_df["T_K"].min())
		candidates = liq_df[np.isclose(liq_df["T_K"], tmin, rtol=1e-12, atol=1e-12)].copy()

		if not candidates.empty:
			candidates = (
				candidates
				.sort_values(by=["T_K", "G"] + self.composition_cols, ascending=True)
				.drop_duplicates(subset=self.composition_cols, keep="first")
				.reset_index(drop=True)
			)

		if candidates.empty:
			return {
				"tmin": tmin,
				"cluster_records": pd.DataFrame(),
				"coexisting_solids_union": [],
			}

		threshold = _infer_grid_delta(self.equilibrium_df, self.composition_cols)
		comp_points = candidates[self.composition_cols].to_numpy(dtype=float)
		components = _connected_components(comp_points, threshold=threshold)

		records = []
		union_solids = set()

		for cluster_id, component in enumerate(components):
			cluster_df = candidates.iloc[component].copy()
			cluster_df = cluster_df.sort_values(by=["T_K", "G"] + self.composition_cols)
			rep = cluster_df.iloc[0]

			indep_points = cluster_df[self.composition_cols].to_numpy(dtype=float)
			full_points = [
				[_round_sig(float(1.0 - np.sum(row)))] + [_round_sig(float(v)) for v in row]
				for row in indep_points
			]

			simplex_ids = sorted(cluster_df["simplex_id"].astype(int).unique().tolist())
			mask = self.equilibrium_df["simplex_id"].isin(simplex_ids)
			phases = self.equilibrium_df.loc[mask, "Phase"].astype(str).tolist()
			coexisting_solids = sorted({p for p in phases if not _phase_is_liquid(p, self.liquid_aliases)})
			union_solids.update(coexisting_solids)

			record = {
				"cluster_id": int(cluster_id),
				"T_K": float(rep["T_K"]),
				"T_C": float(rep.get("T_C", rep["T_K"] - 273.15)),
				"G": float(rep["G"]),
				"n_points": int(len(cluster_df)),
				"simplex_ids": simplex_ids,
				"coexisting_solids": coexisting_solids,
				"cluster_points_independent": [[_round_sig(float(v)) for v in row] for row in indep_points],
				"cluster_points_full": full_points,
			}
			for col in self.composition_cols:
				record[col] = float(rep[col])
			records.append(record)

		records_df = pd.DataFrame(records)
		if not records_df.empty:
			records_df = records_df.sort_values(by=["T_K", "G"] + self.composition_cols).reset_index(drop=True)

		return {
			"tmin": tmin,
			"cluster_records": records_df,
			"coexisting_solids_union": sorted(union_solids),
		}

	def load_and_stitch(self) -> "GeneralHSXPostprocess":
		manifest_paths = self._discover_manifest_paths()
		blocks = [self._build_block(p) for p in manifest_paths]
		blocks = sorted(blocks, key=lambda b: (b.t_min, b.t_max, b.stem))

		self._validate_temp_ranges(blocks)
		self.blocks = blocks

		eq_frames: List[pd.DataFrame] = []
		sx_frames: List[pd.DataFrame] = []
		gtx_frames: List[pd.DataFrame] = []
		block_rows = []

		simplex_offset = 0
		source_offset = 0

		for block_idx, block in enumerate(blocks):
			eq_df = pd.read_pickle(block.equilibrium_path, compression="gzip")
			sx_df = pd.read_pickle(block.simplex_path, compression="gzip")
			gtx_df = pd.read_pickle(block.gtx_path, compression="gzip") if (self.load_gtx and block.gtx_path) else None

			if not self.composition_cols:
				self.composition_cols = _infer_comp_cols(eq_df)

			if "simplex_id" not in eq_df.columns or "source_row_id" not in eq_df.columns:
				raise ValueError(f"Equilibrium file missing linkage columns: {block.equilibrium_path}")
			if "simplex_id" not in sx_df.columns or "vertex_source_row_ids" not in sx_df.columns:
				raise ValueError(f"Simplex file missing linkage columns: {block.simplex_path}")

			eq_df = eq_df.copy()
			sx_df = sx_df.copy()

			eq_df["simplex_id"] = eq_df["simplex_id"].astype(np.int64) + int(simplex_offset)
			sx_df["simplex_id"] = sx_df["simplex_id"].astype(np.int64) + int(simplex_offset)

			eq_df["source_row_id"] = eq_df["source_row_id"].astype(np.int64) + int(source_offset)
			sx_df["vertex_source_row_ids"] = sx_df["vertex_source_row_ids"].apply(
				lambda vals: self._offset_vertex_source_ids(vals, source_offset)
			)

			eq_df["block_index"] = int(block_idx)
			sx_df["block_index"] = int(block_idx)
			if gtx_df is not None:
				gtx_df = gtx_df.copy()
				gtx_df["source_row_id"] = np.arange(len(gtx_df), dtype=np.int64) + int(source_offset)
				gtx_df["block_index"] = int(block_idx)

			eq_frames.append(eq_df)
			sx_frames.append(sx_df)
			if gtx_df is not None:
				gtx_frames.append(gtx_df)

			block_rows.append({
				"block_index": int(block_idx),
				"stem": block.stem,
				"path": str(block.manifest_path.parent),
				"t_min": float(block.t_min),
				"t_max": float(block.t_max),
				"n_gtx_rows": int(block.n_gtx_rows),
				"n_equilibrium_rows": int(block.n_equilibrium_rows),
				"n_simplex_rows": int(block.n_simplex_rows),
				"simplex_offset": int(simplex_offset),
				"source_row_offset": int(source_offset),
			})

			simplex_offset += int(block.n_simplex_rows)
			source_offset += int(block.n_gtx_rows)

		self.equilibrium_df = pd.concat(eq_frames, ignore_index=True) if eq_frames else pd.DataFrame()
		self.simplex_df = pd.concat(sx_frames, ignore_index=True) if sx_frames else pd.DataFrame()
		self.gtx_df = pd.concat(gtx_frames, ignore_index=True) if gtx_frames else pd.DataFrame()
		self.block_index_df = pd.DataFrame(block_rows)

		if self.blocks:
			self.system_name = self.blocks[0].stem.split("_T", 1)[0]

		self.global_eutectic = self._compute_global_eutectic(interior_tol=1e-6)
		self.low_t_clusters = self._compute_low_t_clusters()

		print(
			f"Loaded {len(self.blocks)} block(s) for system {self.system_name or 'unknown'} | "
			f"equilibrium rows={len(self.equilibrium_df)} | "
			f"simplex rows={len(self.simplex_df)} | "
			f"gtx rows={len(self.gtx_df)}"
		)
		if self.block_index_df is not None and not self.block_index_df.empty:
			t_min = float(self.block_index_df["t_min"].min())
			t_max = float(self.block_index_df["t_max"].max())
			print(f"Temperature coverage: {t_min:.2f} K to {t_max:.2f} K")

		return self

	def get_temperature_slice(self, temp_k: float, atol: float = 1e-9) -> pd.DataFrame:
		if self.equilibrium_df.empty:
			return pd.DataFrame(columns=self.equilibrium_df.columns)
		mask = np.isclose(self.equilibrium_df["T_K"].to_numpy(dtype=float), float(temp_k), atol=atol, rtol=0.0)
		return self.equilibrium_df.loc[mask].copy().reset_index(drop=True)

	def get_phase_subset(self, phases: Sequence[str]) -> pd.DataFrame:
		if self.equilibrium_df.empty:
			return pd.DataFrame(columns=self.equilibrium_df.columns)
		wanted = {str(p) for p in phases}
		return self.equilibrium_df[self.equilibrium_df["Phase"].astype(str).isin(wanted)].copy().reset_index(drop=True)

	def get_simplex_vertices(self, simplex_id: int) -> pd.DataFrame:
		if self.equilibrium_df.empty:
			return pd.DataFrame(columns=self.equilibrium_df.columns)
		return (
			self.equilibrium_df[self.equilibrium_df["simplex_id"].astype(int) == int(simplex_id)]
			.copy()
			.reset_index(drop=True)
		)

	def get_source_rows(self, source_row_ids: Sequence[int]) -> pd.DataFrame:
		if self.gtx_df.empty:
			raise ValueError("GTX dataframe is empty. Initialize with load_gtx=True.")

		idx = np.asarray(list(source_row_ids), dtype=np.int64)
		if len(idx) == 0:
			return pd.DataFrame(columns=self.gtx_df.columns)
		if idx.min() < 0 or idx.max() >= len(self.gtx_df):
			raise IndexError("source_row_ids out of stitched GTX bounds.")
		return self.gtx_df.iloc[idx].copy().reset_index(drop=True)

	def _default_plotter_dir(self) -> Path:
		"""Infer quaternary_demo/plotter_dir from the system cache directory."""
		for p in [self.system_cache_dir] + list(self.system_cache_dir.parents):
			if p.name == "lower_hull_cache":
				return p.parent / "plotter_dir"
		return self.system_cache_dir / "plotter_dir"

	def extract_phase_boundary_equilibrium(
		self,
		min_unique_phases: int = 2,
	) -> pd.DataFrame:
		"""Keep only equilibrium rows from simplices spanning >= min_unique_phases.

		This removes simplex rows that belong to a single phase only (e.g. all-L or
		all-solid-solution simplex vertices), which are non-boundary simplices for
		visualization workflows.
		"""
		if self.equilibrium_df.empty:
			self.phase_boundary_equilibrium_df = pd.DataFrame(columns=self.equilibrium_df.columns)
			return self.phase_boundary_equilibrium_df

		if "simplex_id" not in self.equilibrium_df.columns or "Phase" not in self.equilibrium_df.columns:
			raise ValueError("equilibrium_df must contain simplex_id and Phase columns.")

		phase_pairs = self.equilibrium_df[["simplex_id", "Phase"]].drop_duplicates()
		phase_counts = phase_pairs.groupby("simplex_id").size()
		boundary_ids = phase_counts[phase_counts >= int(min_unique_phases)].index

		keep_mask = self.equilibrium_df["simplex_id"].isin(boundary_ids)
		self.phase_boundary_equilibrium_df = self.equilibrium_df.loc[keep_mask].copy().reset_index(drop=True)
		return self.phase_boundary_equilibrium_df

	def save_phase_boundary_equilibrium(
		self,
		output_dir: Optional[str] = None,
		filename: Optional[str] = None,
		compression: str = "gzip",
	) -> Path:
		"""Save phase-boundary-only equilibrium rows for downstream plotting."""
		if self.phase_boundary_equilibrium_df.empty:
			self.extract_phase_boundary_equilibrium(min_unique_phases=2)

		if self.phase_boundary_equilibrium_df.empty:
			raise ValueError("No phase-boundary equilibrium rows available to save.")

		if output_dir is None:
			out_dir = self._default_plotter_dir()
		else:
			out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		if filename is None:
			t_min = float(self.phase_boundary_equilibrium_df["T_K"].min()) if "T_K" in self.phase_boundary_equilibrium_df.columns else 0.0
			t_max = float(self.phase_boundary_equilibrium_df["T_K"].max()) if "T_K" in self.phase_boundary_equilibrium_df.columns else 0.0
			filename = f"{self.system_name or 'system'}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium.pkl.gz"

		cols = [c for c in self.composition_cols + ["T_K", "T_C", "G", "Phase", "simplex_id", "source_row_id"] if c in self.phase_boundary_equilibrium_df.columns]
		to_save = self.phase_boundary_equilibrium_df[cols].copy()

		out_path = out_dir / filename
		to_save.to_pickle(out_path, compression=compression)
		self.phase_boundary_equilibrium_path = out_path
		return out_path

	def evaluate_tmax_filter_change(self, atol: float = 1e-9) -> Dict[str, Any]:
		"""Compare Tmax before/after boundary filtering and inspect top-T phases."""
		if self.equilibrium_df.empty:
			d = {
				"full_tmax_k": None,
				"filtered_tmax_k": None,
				"tmax_changed_after_filter": None,
				"all_phases_at_full_tmax_are_liquid": None,
				"phases_at_full_tmax": [],
			}
			self.tmax_filter_diagnostic = d
			return d

		if self.phase_boundary_equilibrium_df.empty:
			self.extract_phase_boundary_equilibrium(min_unique_phases=2)

		full_tmax = float(self.equilibrium_df["T_K"].max())
		if self.phase_boundary_equilibrium_df.empty:
			filtered_tmax = None
			tmax_changed = True
		else:
			filtered_tmax = float(self.phase_boundary_equilibrium_df["T_K"].max())
			tmax_changed = not bool(np.isclose(full_tmax, filtered_tmax, atol=atol, rtol=0.0))

		top_mask = np.isclose(self.equilibrium_df["T_K"].to_numpy(dtype=float), full_tmax, atol=atol, rtol=0.0)
		top_rows = self.equilibrium_df.loc[top_mask]
		phases_at_top = sorted(top_rows["Phase"].astype(str).unique().tolist()) if not top_rows.empty else []
		all_liquid = bool(len(phases_at_top) > 0 and all(_phase_is_liquid(p, self.liquid_aliases) for p in phases_at_top))

		d = {
			"full_tmax_k": full_tmax,
			"filtered_tmax_k": filtered_tmax,
			"tmax_changed_after_filter": bool(tmax_changed),
			"all_phases_at_full_tmax_are_liquid": all_liquid,
			"phases_at_full_tmax": phases_at_top,
		}
		self.tmax_filter_diagnostic = d
		return d

	def summary(self) -> Dict[str, Any]:
		t_min = float(self.equilibrium_df["T_K"].min()) if not self.equilibrium_df.empty else None
		t_max = float(self.equilibrium_df["T_K"].max()) if not self.equilibrium_df.empty else None
		return {
			"system_name": self.system_name,
			"system_cache_dir": str(self.system_cache_dir),
			"n_blocks": int(len(self.blocks)),
			"n_equilibrium_rows": int(len(self.equilibrium_df)),
			"n_simplex_rows": int(len(self.simplex_df)),
			"n_gtx_rows": int(len(self.gtx_df)),
			"composition_cols": list(self.composition_cols),
			"temp_range": [t_min, t_max],
			"global_eutectic": self.global_eutectic,
			"low_t_cluster_count": int(len(self.low_t_clusters.get("cluster_records", [])))
			if isinstance(self.low_t_clusters.get("cluster_records"), pd.DataFrame)
			else 0,
			"n_phase_boundary_rows": int(len(self.phase_boundary_equilibrium_df)),
			"phase_boundary_equilibrium_path": str(self.phase_boundary_equilibrium_path) if self.phase_boundary_equilibrium_path else None,
			"tmax_filter_diagnostic": self.tmax_filter_diagnostic,
		}


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


def _parse_int_env(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return int(default)
	return int(raw)


def _parse_float_env(name: str, default: float) -> float:
	raw = os.getenv(name)
	if raw is None:
		return float(default)
	return float(raw)


def main_post() -> None:
	default_dir = "all_dumps/quaternary_demo/lower_hull_cache/Hf-Nb-W-Zr"
	system_dir = os.getenv("GP_SYSTEM_CACHE_DIR", default_dir)
	recursive = _parse_bool_env("GP_RECURSIVE", True)
	load_gtx = _parse_bool_env("GP_LOAD_GTX", True)
	load_and_stitch = _parse_bool_env("GP_LOAD_AND_STITCH", True)
	extract_phase_boundary = _parse_bool_env("GP_EXTRACT_PHASE_BOUNDARY", True)
	save_phase_boundary = _parse_bool_env("GP_SAVE_PHASE_BOUNDARY", True)
	notify_tmax_unchanged = _parse_bool_env("GP_NOTIFY_TMAX_UNCHANGED", True)
	tmax_change_atol = _parse_float_env("GP_TMAX_CHANGE_ATOL", 1e-9)
	plotter_dir_override = os.getenv("GP_PLOTTER_DIR")
	print_rows = _parse_int_env("GP_PRINT_ROWS", 20)

	print("=" * 72)
	print("GENERAL PLOTTER POSTPROCESS SMOKE TEST")
	print("=" * 72)
	print(f"System cache dir: {system_dir}")
	print(f"Recursive manifest discovery: {recursive}")
	print(f"Load GTX during stitch: {load_gtx}")
	print(f"Execute full load_and_stitch: {load_and_stitch}")
	print(f"Extract phase-boundary simplices: {extract_phase_boundary}")
	print(f"Save phase-boundary dataframe: {save_phase_boundary}")
	print(f"Notify when Tmax unchanged after filter: {notify_tmax_unchanged}")
	print(f"Tmax change tolerance: {tmax_change_atol}")
	print(f"Plotter dir override: {plotter_dir_override}")
	print(f"Rows to print from reconstructed equilibrium_df: {print_rows}")

	post = GeneralHSXPostprocess(
		system_cache_dir=system_dir,
		recursive=recursive,
		load_gtx=load_gtx,
	)

	print("\n[Init] Object attributes:")
	print(f"  system_cache_dir: {post.system_cache_dir}")
	print(f"  recursive: {post.recursive}")
	print(f"  load_gtx: {post.load_gtx}")
	print(f"  liquid_aliases: {post.liquid_aliases}")
	print(f"  equilibrium_df (pre-stitch) shape: {post.equilibrium_df.shape}")

	manifests = post._discover_manifest_paths()
	print(f"\n[Discovery] manifests found: {len(manifests)}")
	for i, path in enumerate(manifests[:10]):
		print(f"  {i}: {path}")
	if len(manifests) > 10:
		print(f"  ... ({len(manifests) - 10} more)")

	blocks = [post._build_block(p) for p in manifests]
	blocks = sorted(blocks, key=lambda b: (b.t_min, b.t_max, b.stem))
	post._validate_temp_ranges(blocks)

	print("\n[Blocks] parsed metadata:")
	for i, b in enumerate(blocks):
		print(
			f"  {i}: stem={b.stem} | T=[{b.t_min:.2f}, {b.t_max:.2f}] K | "
			f"n_eq={b.n_equilibrium_rows} | n_sx={b.n_simplex_rows} | n_gtx={b.n_gtx_rows}"
		)

	if load_and_stitch:
		print("\n[Stitch] Running load_and_stitch()...")
		post.load_and_stitch()
		print("\n[DataFrames]")
		print(f"  equilibrium_df shape: {post.equilibrium_df.shape}")
		print(f"  simplex_df shape: {post.simplex_df.shape}")
		print(f"  gtx_df shape: {post.gtx_df.shape}")

		if not post.equilibrium_df.empty:
			show_cols = [c for c in post.composition_cols + ["T_K", "T_C", "G", "Phase", "simplex_id", "source_row_id"] if c in post.equilibrium_df.columns]
			print("\n[Reconstructed equilibrium_df sample]")
			print(post.equilibrium_df[show_cols].head(max(1, print_rows)))

		if extract_phase_boundary:
			print("\n[Phase boundary filter] Extracting boundary simplices (>=2 unique phases per simplex_id)...")
			phase_boundary_df = post.extract_phase_boundary_equilibrium(min_unique_phases=2)
			print(f"  phase_boundary_equilibrium_df shape: {phase_boundary_df.shape}")
			if not phase_boundary_df.empty:
				show_cols_pb = [c for c in post.composition_cols + ["T_K", "T_C", "G", "Phase", "simplex_id", "source_row_id"] if c in phase_boundary_df.columns]
				print("\n[Phase-boundary equilibrium sample]")
				print(phase_boundary_df[show_cols_pb].head(max(1, print_rows)))

			if save_phase_boundary:
				out_path = post.save_phase_boundary_equilibrium(output_dir=plotter_dir_override)
				print(f"Saved phase-boundary equilibrium file: {out_path}")

			tmax_diag = post.evaluate_tmax_filter_change(atol=tmax_change_atol)
			print("\n[Tmax diagnostic]")
			print(
				f"  full_tmax_k={tmax_diag['full_tmax_k']} | "
				f"filtered_tmax_k={tmax_diag['filtered_tmax_k']} | "
				f"tmax_changed_after_filter={tmax_diag['tmax_changed_after_filter']}"
			)
			print(f"  all_phases_at_full_tmax_are_liquid={tmax_diag['all_phases_at_full_tmax_are_liquid']}")
			print(f"  phases_at_full_tmax={tmax_diag['phases_at_full_tmax']}")

			if notify_tmax_unchanged and (tmax_diag.get("tmax_changed_after_filter") is False):
				print(
					"[WARN] Boundary-filter Tmax did not decrease. "
					"Highest-temperature rows still include non-single-phase boundary simplices; "
					"consider sampling a higher Tmax range to ensure complete high-T closure."
				)

		s = post.summary()
		print("\n[Summary]")
		print(json.dumps(s, indent=2, default=str))
	else:
		print("\n[Stitch] Skipped. Set GP_LOAD_AND_STITCH=true to run full stitching.")

	print("=" * 72)


if __name__ == "__main__":
	main_post()


