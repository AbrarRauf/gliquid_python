"""Post-processing helpers for stitched General lower-hull cache artifacts."""

from __future__ import annotations

import json
import os
import re
import fnmatch
import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import Delaunay, QhullError, cKDTree


SS_FIXED_COLORS = {
	"BCC": "#d7263d",
	"FCC": "#1b9aaa",
	"HCP": "#f4a259",
}


def _read_pickle_robust(path: str) -> pd.DataFrame:
	try:
		return pd.read_pickle(path, compression="gzip")
	except NotImplementedError as exc:
		if "NDArrayBacked" not in str(exc) and "array(" not in str(exc):
			raise
		warnings.warn(
			f"Retrying legacy-compatible pickle load for {path}: {exc}",
			RuntimeWarning,
		)

		@contextlib.contextmanager
		def _legacy_stringarray_setstate_patch() -> Any:
			from pandas.core.arrays.string_ import StringArray

			orig_setstate = StringArray.__setstate__

			def _patched_setstate(self, state):
				if isinstance(state, tuple) and len(state) == 2:
					dtype_obj, values = state
					if isinstance(dtype_obj, str) and dtype_obj.lower() in {"str", "string"}:
						dtype_obj = pd.StringDtype(storage="python")
					state = (dtype_obj, values, {})
				return orig_setstate(self, state)

			StringArray.__setstate__ = _patched_setstate
			try:
				yield
			finally:
				StringArray.__setstate__ = orig_setstate

		with _legacy_stringarray_setstate_patch():
			return pd.read_pickle(path, compression="gzip")


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


class GeneralPostProcess:
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
	) -> "GeneralPostProcess":
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

	def load_and_stitch(self) -> "GeneralPostProcess":
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
			eq_df = _read_pickle_robust(block.equilibrium_path)
			sx_df = _read_pickle_robust(block.simplex_path)
			gtx_df = _read_pickle_robust(block.gtx_path) if (self.load_gtx and block.gtx_path) else None

			if (not self.composition_cols) and (not eq_df.empty):
				self.composition_cols = _infer_comp_cols(eq_df)

			if (not eq_df.empty) and ("simplex_id" not in eq_df.columns or "source_row_id" not in eq_df.columns):
				raise ValueError(f"Equilibrium file missing linkage columns: {block.equilibrium_path}")
			if (not sx_df.empty) and ("simplex_id" not in sx_df.columns or "vertex_source_row_ids" not in sx_df.columns):
				raise ValueError(f"Simplex file missing linkage columns: {block.simplex_path}")

			eq_df = eq_df.copy()
			sx_df = sx_df.copy()

			if "simplex_id" in eq_df.columns:
				eq_df["simplex_id"] = eq_df["simplex_id"].astype(np.int64) + int(simplex_offset)
			if "simplex_id" in sx_df.columns:
				sx_df["simplex_id"] = sx_df["simplex_id"].astype(np.int64) + int(simplex_offset)

			if "source_row_id" in eq_df.columns:
				eq_df["source_row_id"] = eq_df["source_row_id"].astype(np.int64) + int(source_offset)
			if "vertex_source_row_ids" in sx_df.columns:
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

	def save_phase_boundary_sample_csv(
		self,
		output_dir: Optional[str] = None,
		filename: Optional[str] = None,
		max_rows: int = 10000,
		n_temp_bins: int = 24,
		random_state: int = 42,
		dump_full: bool = False,
	) -> Path:
		"""Save a stratified sample CSV snapshot of phase-boundary equilibrium rows.

		Sampling targets broad coverage across phases and temperature bins while
		respecting a maximum row count unless dump_full=True.
		"""
		if self.phase_boundary_equilibrium_df.empty:
			self.extract_phase_boundary_equilibrium(min_unique_phases=2)

		if self.phase_boundary_equilibrium_df.empty:
			raise ValueError("No phase-boundary equilibrium rows available to sample.")

		if output_dir is None:
			out_dir = self._default_plotter_dir()
		else:
			out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

		df = self.phase_boundary_equilibrium_df.copy()
		temp_col = "T_K" if "T_K" in df.columns else None
		phase_col = "Phase" if "Phase" in df.columns else None

		dump_full = bool(dump_full)
		max_rows = int(max(1, max_rows))
		n_temp_bins = int(max(2, n_temp_bins))

		if (not dump_full) and len(df) > max_rows and temp_col is not None and phase_col is not None:
			work = df.copy()
			n_unique_t = int(work[temp_col].nunique())
			q = int(max(2, min(n_temp_bins, n_unique_t))) if n_unique_t > 0 else 2

			try:
				work["_temp_bin"] = pd.qcut(work[temp_col].astype(float), q=q, duplicates="drop")
			except Exception:
				work["_temp_bin"] = "all"

			groups = work.groupby([phase_col, "_temp_bin"], observed=True, sort=False)
			n_groups = max(1, int(groups.ngroups))
			per_group = max(1, max_rows // n_groups)

			parts = []
			for _, g in groups:
				n_take = min(len(g), per_group)
				parts.append(g.sample(n=n_take, random_state=random_state))

			sampled = pd.concat(parts, axis=0) if parts else work.iloc[0:0].copy()

			if len(sampled) < max_rows:
				remaining = work.drop(index=sampled.index, errors="ignore")
				need = max_rows - len(sampled)
				if len(remaining) > 0 and need > 0:
					extra = remaining.sample(n=min(need, len(remaining)), random_state=random_state)
					sampled = pd.concat([sampled, extra], axis=0)

			if len(sampled) > max_rows:
				sampled = sampled.sample(n=max_rows, random_state=random_state)

			df_out = sampled.drop(columns=["_temp_bin"], errors="ignore")
			if temp_col in df_out.columns and phase_col in df_out.columns:
				df_out = df_out.sort_values(by=[phase_col, temp_col]).reset_index(drop=True)
			else:
				df_out = df_out.reset_index(drop=True)
		else:
			df_out = df.copy().reset_index(drop=True)

		if filename is None:
			t_min = float(df["T_K"].min()) if "T_K" in df.columns else 0.0
			t_max = float(df["T_K"].max()) if "T_K" in df.columns else 0.0
			if dump_full:
				filename = f"{self.system_name or 'system'}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium_full.csv"
			else:
				filename = f"{self.system_name or 'system'}_T{t_min:.2f}_{t_max:.2f}_phase_boundary_equilibrium_sample.csv"

		out_path = out_dir / filename
		df_out.to_csv(out_path, index=False)
		return out_path

	def evaluate_tmax_filter_change(self, atol: float = 1e-9) -> Dict[str, Any]:
		"""Evaluate temperature-boundary diagnostics on stitched equilibrium data.

		Includes:
		- Tmax before/after phase-boundary filtering.
		- Whether full Tmax slice is entirely liquid.
		- Whether full Tmin slice still contains any liquid (indicates lower-T
		  extension may be needed).
		"""
		if self.equilibrium_df.empty:
			d = {
				"full_tmax_k": None,
				"filtered_tmax_k": None,
				"tmax_changed_after_filter": None,
				"all_phases_at_full_tmax_are_liquid": None,
				"phases_at_full_tmax": [],
				"full_tmin_k": None,
				"liquid_present_at_full_tmin": None,
				"phases_at_full_tmin": [],
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

		full_tmin = float(self.equilibrium_df["T_K"].min())
		bot_mask = np.isclose(self.equilibrium_df["T_K"].to_numpy(dtype=float), full_tmin, atol=atol, rtol=0.0)
		bot_rows = self.equilibrium_df.loc[bot_mask]
		phases_at_bottom = sorted(bot_rows["Phase"].astype(str).unique().tolist()) if not bot_rows.empty else []
		liquid_present_at_bottom = bool(any(_phase_is_liquid(p, self.liquid_aliases) for p in phases_at_bottom))

		d = {
			"full_tmax_k": full_tmax,
			"filtered_tmax_k": filtered_tmax,
			"tmax_changed_after_filter": bool(tmax_changed),
			"all_phases_at_full_tmax_are_liquid": all_liquid,
			"phases_at_full_tmax": phases_at_top,
			"full_tmin_k": full_tmin,
			"liquid_present_at_full_tmin": liquid_present_at_bottom,
			"phases_at_full_tmin": phases_at_bottom,
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


class PhaseBoundaryPlotter:
	"""Slice and plot stitched phase-boundary equilibrium data.

	This class assumes the reduced data format produced by
	GeneralPostProcess.save_phase_boundary_equilibrium(...).
	"""

	def __init__(
		self,
		equilibrium_df: pd.DataFrame,
		composition_cols: Optional[Sequence[str]] = None,
		element_names: Optional[Sequence[str]] = None,
		liquid_aliases: Sequence[str] = ("L", "LIQUID"),
		phase_colors: Optional[Dict[str, str]] = None,
		solution_phase_pattern: str = r"(FCC|BCC|HCP|A1|A2|A3|A4|SOLUTION|_SS|\bSS\b)",
	):
		self.equilibrium_df = equilibrium_df.copy().reset_index(drop=True)
		self.composition_cols = list(composition_cols) if composition_cols else _infer_comp_cols(self.equilibrium_df)
		self.liquid_aliases = tuple(liquid_aliases)
		self.solution_phase_re = re.compile(solution_phase_pattern, flags=re.IGNORECASE)
		self.phase_colors = dict(phase_colors or {})

		self.n_indep_components = len(self.composition_cols)
		self.n_total_components = self.n_indep_components + 1

		self.element_names = list(element_names) if element_names is not None else None
		if self.element_names is not None and len(self.element_names) != self.n_total_components:
			raise ValueError(
				"element_names length must equal total component count "
				f"({self.n_total_components}), got {len(self.element_names)}"
			)

		validation = self.validate_schema()
		if not validation["valid"]:
			raise ValueError("Invalid equilibrium dataframe schema: " + "; ".join(validation["errors"]))

		self._df_full = self._prepare_full_composition_df(self.equilibrium_df)
		self._full_comp_cols = ["x_dep"] + self.composition_cols
		self._simplex_phase_map = self._build_simplex_phase_map()

	def validate_schema(self) -> Dict[str, Any]:
		errors: List[str] = []
		required_cols = ["Phase", "T_K"] + self.composition_cols
		missing = [c for c in required_cols if c not in self.equilibrium_df.columns]
		if missing:
			errors.append(f"Missing required columns: {missing}")

		if self.equilibrium_df.empty:
			errors.append("equilibrium_df is empty")

		for col in self.composition_cols + ["T_K"]:
			if col in self.equilibrium_df.columns:
				if not np.issubdtype(self.equilibrium_df[col].dtype, np.number):
					errors.append(f"Column {col} must be numeric")

		return {"valid": len(errors) == 0, "errors": errors}

	def validate_composition_integrity(self, tolerance: float = 1e-6) -> Dict[str, Any]:
		x_dep = 1.0 - self.equilibrium_df[self.composition_cols].sum(axis=1).to_numpy(dtype=float)
		invalid_mask = (x_dep < -float(tolerance)) | (x_dep > 1.0 + float(tolerance))
		return {
			"n_rows": int(len(self.equilibrium_df)),
			"n_invalid_rows": int(np.sum(invalid_mask)),
			"min_x_dep": float(np.min(x_dep)) if len(x_dep) else None,
			"max_x_dep": float(np.max(x_dep)) if len(x_dep) else None,
		}

	def _prepare_full_composition_df(self, df: pd.DataFrame) -> pd.DataFrame:
		out = df.copy()
		x_dep = 1.0 - out[self.composition_cols].sum(axis=1)
		out["x_dep"] = x_dep.astype(float)
		return out

	def _build_simplex_phase_map(self) -> Dict[int, Tuple[str, ...]]:
		"""Build simplex_id -> sorted unique phases once for efficient hover enrichment."""
		if "simplex_id" not in self.equilibrium_df.columns:
			return {}
		simplex_phase_series = (
			self.equilibrium_df[["simplex_id", "Phase"]]
			.dropna(subset=["simplex_id", "Phase"])
			.groupby("simplex_id")["Phase"]
			.apply(lambda s: tuple(sorted({str(v) for v in s.tolist()})))
		)
		return {int(k): tuple(v) for k, v in simplex_phase_series.items()}

	def _attach_coexisting_phases(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Add coexisting phase list per row using simplex cache; excludes the row phase itself."""
		out = df.copy()
		if "simplex_id" not in out.columns or not self._simplex_phase_map:
			out["coexisting_phases"] = ""
			return out

		simplex_vals = out["simplex_id"].to_numpy()
		phase_vals = out["Phase"].astype(str).to_numpy()
		coexist = []
		for sid, ph in zip(simplex_vals, phase_vals):
			if pd.isna(sid):
				coexist.append("")
				continue
			phases = self._simplex_phase_map.get(int(sid), tuple())
			others = [p for p in phases if p != ph]
			coexist.append(", ".join(others))
		out["coexisting_phases"] = coexist
		return out

	def _component_label(self, idx: int) -> str:
		if self.element_names is not None:
			return str(self.element_names[idx])
		if idx == 0:
			return "dep"
		return f"x{idx - 1}"

	def _phase_is_solution(self, phase: Any) -> bool:
		p = str(phase)
		if _phase_is_liquid(p, self.liquid_aliases):
			return False
		return self.solution_phase_re.search(p) is not None

	def _ensure_quaternary(self):
		if self.n_total_components != 4:
			raise ValueError(
				f"Quaternary plotting requires 4 total components, got {self.n_total_components}"
			)

	def _resolve_tol(self, tol: Optional[float]) -> float:
		if tol is not None:
			return float(max(0.0, tol))
		base = _infer_grid_delta(self.equilibrium_df, self.composition_cols)
		if base > 0.0:
			return float(max(base * 0.5, 1e-6))
		return 1e-6

	def _quantize_composition_for_grouping(self, df: pd.DataFrame, tol: float) -> pd.Series:
		t = self._resolve_tol(tol)
		arr = df[self._full_comp_cols].to_numpy(dtype=float)
		q = np.round(arr / t).astype(np.int64)
		keys = [tuple(row.tolist()) for row in q]
		return pd.Series(keys, index=df.index)

	def get_phase_list(self, include_liquid: bool = True) -> List[str]:
		phases = self.equilibrium_df["Phase"].astype(str)
		if include_liquid:
			return sorted(phases.unique().tolist())
		return sorted([p for p in phases.unique().tolist() if not _phase_is_liquid(p, self.liquid_aliases)])

	def get_temperature_statistics(self) -> Dict[str, Any]:
		t = self.equilibrium_df["T_K"].to_numpy(dtype=float)
		return {
			"T_min_K": float(np.min(t)) if len(t) else None,
			"T_max_K": float(np.max(t)) if len(t) else None,
			"n_unique_T": int(self.equilibrium_df["T_K"].nunique()),
		}

	def _full_component_names(self) -> List[str]:
		if self.element_names is not None:
			return [str(v) for v in self.element_names]
		# Generic fallback for arbitrary component count: dep + x0..x{n-2}
		return ["dep"] + [f"x{i}" for i in range(self.n_total_components - 1)]

	def _phase_color_map(self, phases: Sequence[str]) -> Dict[str, str]:
		reserved = {"cornflowerblue"}
		reserved.update(SS_FIXED_COLORS.values())
		palette = [c for c in px.colors.qualitative.Dark24 if c not in reserved]
		if not palette:
			palette = px.colors.qualitative.Dark24

		out = dict(self.phase_colors)
		used = set()
		next_i = 0

		for p in sorted({str(v) for v in phases}):
			p_str = str(p)
			p_up = p_str.upper()

			# Hard conventions: liquid and SS families are fixed.
			if _phase_is_liquid(p_str, self.liquid_aliases):
				out[p_str] = "cornflowerblue"
				used.add("cornflowerblue")
				continue
			if "BCC" in p_up:
				out[p_str] = SS_FIXED_COLORS["BCC"]
				used.add(SS_FIXED_COLORS["BCC"])
				continue
			if "FCC" in p_up:
				out[p_str] = SS_FIXED_COLORS["FCC"]
				used.add(SS_FIXED_COLORS["FCC"])
				continue
			if "HCP" in p_up:
				out[p_str] = SS_FIXED_COLORS["HCP"]
				used.add(SS_FIXED_COLORS["HCP"])
				continue

			# Preserve explicit user colors for intermetallics if they do not violate reserved colors.
			if p_str in out and out[p_str] not in reserved:
				used.add(out[p_str])
				continue

			while next_i < len(palette) and palette[next_i] in used:
				next_i += 1
			if next_i >= len(palette):
				next_i = 0
			out[p_str] = palette[next_i]
			used.add(palette[next_i])
			next_i += 1
		return out

	def _full_component_df(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
		base = self._df_full if df is None else self._prepare_full_composition_df(df)
		name_map = dict(zip(self._full_comp_cols, self._full_component_names()))
		out = base.copy()
		for src, dst in name_map.items():
			out[dst] = out[src].astype(float)
		return out

	@staticmethod
	def _snap_to_grid(value: float, grid_delta: float) -> float:
		if grid_delta <= 0.0:
			return float(value)
		return float(np.round(float(value) / grid_delta) * grid_delta)

	@staticmethod
	def _cartesian_to_ternary_display(x_vals: np.ndarray, y_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Match ternary_HSX cartesian_to_ternary transform for display coordinates."""
		tx = x_vals + 0.5 * y_vals
		ty = (np.sqrt(3.0) / 2.0) * y_vals
		return tx, ty

	@staticmethod
	def _ternary_display_to_cartesian(tx_vals: np.ndarray, ty_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""Inverse of _cartesian_to_ternary_display for hover/reporting in cartesian comps."""
		y = (2.0 / np.sqrt(3.0)) * ty_vals
		x = tx_vals - 0.5 * y
		return x, y

	def _nearest_available_value(self, series: pd.Series, target: float) -> float:
		vals = np.sort(series.dropna().unique().astype(float))
		if len(vals) == 0:
			return float(target)
		idx = int(np.argmin(np.abs(vals - float(target))))
		return float(vals[idx])

	def filter_by_fixed_components(
		self,
		fixed_components: Dict[str, float],
		tolerance: Optional[float] = None,
	) -> pd.DataFrame:
		df = self._full_component_df()
		tol = self._resolve_tol(tolerance)
		mask = np.ones(len(df), dtype=bool)
		for comp_name, target in fixed_components.items():
			if comp_name not in df.columns:
				raise ValueError(f"Unknown component name in fixed_components: {comp_name}")
			mask &= np.abs(df[comp_name].to_numpy(dtype=float) - float(target)) <= tol
		out = df.loc[mask].copy().reset_index(drop=True)
		if out.empty:
			raise ValueError(
				f"No rows found for fixed components {fixed_components} with tolerance {tol}."
			)
		out["is_interpolated"] = False
		return self._attach_coexisting_phases(out)

	def _interpolate_slice_for_plot(
		self,
		axis_components: Sequence[str],
		fixed_components: Dict[str, float],
		tolerance: Optional[float] = None,
		k_nearest: int = 8,
	) -> pd.DataFrame:
		"""Interpolate off-grid constrained slices from nearby grid points.

		Interpolation is done per (phase, quantized axis composition) group using
		inverse-distance weights over fixed-component space.
		"""
		df = self._full_component_df()
		for c in list(axis_components) + list(fixed_components.keys()):
			if c not in df.columns:
				raise ValueError(f"Unknown component for interpolation: {c}")

		tol = self._resolve_tol(tolerance)
		k_nearest = max(2, int(k_nearest))
		rows: List[Dict[str, Any]] = []

		for phase_name, g in df.groupby("Phase", sort=False):
			g_local = g.copy()
			axis_arr = g_local[list(axis_components)].to_numpy(dtype=float)
			q = np.round(axis_arr / tol).astype(np.int64)
			g_local["_axis_key"] = [tuple(v.tolist()) for v in q]

			for _, gg in g_local.groupby("_axis_key", sort=False):
				if len(gg) < 2:
					continue

				d = np.zeros(len(gg), dtype=float)
				for comp_name, target in fixed_components.items():
					d += (gg[comp_name].to_numpy(dtype=float) - float(target)) ** 2
				d = np.sqrt(d)

				order = np.argsort(d)
				sel = order[: min(len(order), k_nearest)]
				gg_sel = gg.iloc[sel]
				d_sel = d[sel]

				weights = 1.0 / np.maximum(d_sel, 1e-12)
				w_sum = float(np.sum(weights))
				if w_sum <= 0.0 or not np.isfinite(w_sum):
					continue
				weights = weights / w_sum

				row: Dict[str, Any] = {"Phase": str(phase_name)}
				for col in ["T_K", "T_C", "G"]:
					if col in gg_sel.columns:
						row[col] = float(np.sum(gg_sel[col].to_numpy(dtype=float) * weights))

				for comp_name in fixed_components:
					row[comp_name] = float(fixed_components[comp_name])

				# Keep plotting-axis compositions from local neighboring points.
				for comp_name in axis_components:
					row[comp_name] = float(np.sum(gg_sel[comp_name].to_numpy(dtype=float) * weights))

				# Preserve extra composition columns when present.
				for comp_name in self._full_component_names():
					if comp_name in row:
						continue
					if comp_name in gg_sel.columns:
						row[comp_name] = float(np.sum(gg_sel[comp_name].to_numpy(dtype=float) * weights))

				row["simplex_id"] = np.nan
				row["source_row_id"] = np.nan
				rows.append(row)

		if not rows:
			return pd.DataFrame()

		out = pd.DataFrame(rows)
		out["is_interpolated"] = True
		return self._attach_coexisting_phases(out)

	def _phase_extrema_mode_for_slice(self, phase: Any) -> Optional[str]:
		"""Return extrema mode for a phase in slice plotting: min for liquid, max for SS."""
		p = str(phase)
		p_up = p.upper()
		if _phase_is_liquid(p, self.liquid_aliases):
			return "min"
		if ("BCC" in p_up) or ("FCC" in p_up) or ("HCP" in p_up):
			return "max"
		return None

	def _reduce_slice_by_phase_extrema(
		self,
		df: pd.DataFrame,
		composition_cols: Sequence[str],
		tol: Optional[float] = None,
	) -> pd.DataFrame:
		"""Apply per-composition phase extrema reduction for binary/ternary slice plots."""
		if df.empty:
			return df

		threshold = self._resolve_tol(tol)
		keep_idx: List[int] = []

		for phase_name, g in df.groupby("Phase", sort=False):
			mode = self._phase_extrema_mode_for_slice(phase_name)
			if mode is None:
				keep_idx.extend(g.index.tolist())
				continue

			g_local = g.copy()
			arr = g_local[list(composition_cols)].to_numpy(dtype=float)
			q = np.round(arr / threshold).astype(np.int64)
			g_local["_qkey"] = [tuple(row.tolist()) for row in q]

			if mode == "min":
				idx = g_local.groupby("_qkey")["T_K"].idxmin()
			else:
				idx = g_local.groupby("_qkey")["T_K"].idxmax()
			keep_idx.extend(idx.astype(int).tolist())

		if not keep_idx:
			return df.iloc[0:0].copy()

		keep_unique = sorted(set(keep_idx))
		return df.loc[keep_unique].copy().reset_index(drop=True)

	def plot_binary_slice_tx(
		self,
		comp_a: str,
		comp_b: str,
		fixed_components: Dict[str, float],
		tolerance: Optional[float] = None,
		phase_extrema_filter: bool = False,
		temp_axis_range_k: Optional[Tuple[float, float]] = None,
		title: Optional[str] = None,
	) -> go.Figure:
		if comp_a in fixed_components or comp_b in fixed_components:
			raise ValueError(
				f"fixed_components cannot include plotted binary components ({comp_a}, {comp_b})."
			)
		try:
			df = self.filter_by_fixed_components(fixed_components=fixed_components, tolerance=tolerance)
		except ValueError:
			df = self._interpolate_slice_for_plot(
				axis_components=[comp_a, comp_b],
				fixed_components=fixed_components,
				tolerance=tolerance,
			)
			if df.empty:
				raise
		if comp_a not in df.columns or comp_b not in df.columns:
			raise ValueError(f"Binary components must exist in dataframe columns: {comp_a}, {comp_b}")

		pair_sum = df[comp_a].to_numpy(dtype=float) + df[comp_b].to_numpy(dtype=float)
		valid = pair_sum > 1e-12
		df = df.loc[valid].copy()
		pair_sum = pair_sum[valid]
		if df.empty:
			raise ValueError(f"No valid rows for binary ratio {comp_a}/({comp_a}+{comp_b}) after filtering.")

		if phase_extrema_filter:
			df = self._reduce_slice_by_phase_extrema(
				df=df,
				composition_cols=[comp_a, comp_b],
				tol=tolerance,
			)
			if df.empty:
				raise ValueError("Phase-extrema filter removed all rows for this binary slice.")
			pair_sum = df[comp_a].to_numpy(dtype=float) + df[comp_b].to_numpy(dtype=float)

		df["x_pair"] = df[comp_a].to_numpy(dtype=float) / pair_sum
		df["pair_total"] = pair_sum
		pair_total_nominal = float(np.median(pair_sum)) if len(pair_sum) else 0.0
		df[f"{comp_a}_abs_min"] = 0.0
		df[f"{comp_a}_abs_max"] = pair_total_nominal
		df[f"{comp_b}_abs_min"] = 0.0
		df[f"{comp_b}_abs_max"] = pair_total_nominal
		if "is_interpolated" not in df.columns:
			df["is_interpolated"] = False
		color_map = self._phase_color_map(df["Phase"].astype(str).tolist())
		fig = px.scatter(
			df,
			x="x_pair",
			y="T_K",
			color="Phase",
			color_discrete_map=color_map,
			title=title or f"Binary Slice {comp_a}-{comp_b}",
			hover_data={
				comp_a: ":.4f",
				comp_b: ":.4f",
				"T_K": ":.2f",
				"x_pair": ":.4f",
					"pair_total": ":.4f",
					f"{comp_a}_abs_min": ":.4f",
					f"{comp_a}_abs_max": ":.4f",
					f"{comp_b}_abs_min": ":.4f",
					f"{comp_b}_abs_max": ":.4f",
						"is_interpolated": True,
					"coexisting_phases": True,
			},
		)
		fig.update_traces(marker=dict(size=6, opacity=0.8))
		fig.update_layout(
			autosize=False,
			width=980,
			height=700,
			xaxis_title=f"x_{comp_a} / (x_{comp_a} + x_{comp_b})",
			yaxis=dict(title="T [K]", range=list(temp_axis_range_k) if temp_axis_range_k is not None else None),
			plot_bgcolor="white",
			legend=dict(x=0.95, y=0.95, xanchor="left", yanchor="top"),
			margin=dict(l=60, r=60, b=60, t=60),
			title=(title or f"Binary Slice {comp_a}-{comp_b}") + f" | in-system bounds: [0, {pair_total_nominal:.3f}]",
		)
		# Draw a full black frame around the binary axes (all four sides).
		fig.update_xaxes(
			showline=True,
			linewidth=2,
			linecolor="black",
			mirror=True,
			ticks="inside",
		)
		fig.update_yaxes(
			showline=True,
			linewidth=2,
			linecolor="black",
			mirror=True,
			ticks="inside",
		)
		return fig

	def plot_ternary_slice(
		self,
		comp_a: str,
		comp_b: str,
		comp_c: str,
		fixed_components: Dict[str, float],
		tolerance: Optional[float] = None,
		phase_extrema_filter: bool = False,
		ternary_phase_mesh: bool = False,
		slice_grid_delta: Optional[float] = None,
		temp_axis_range_k: Optional[Tuple[float, float]] = None,
		title: Optional[str] = None,
		ss_cluster_factor: float = 1.75,
		color_by: str = "Phase",
	) -> go.Figure:
		if comp_a in fixed_components or comp_b in fixed_components or comp_c in fixed_components:
			raise ValueError(
				f"fixed_components cannot include plotted ternary components ({comp_a}, {comp_b}, {comp_c})."
			)
		try:
			df = self.filter_by_fixed_components(fixed_components=fixed_components, tolerance=tolerance)
		except ValueError:
			df = self._interpolate_slice_for_plot(
				axis_components=[comp_a, comp_b, comp_c],
				fixed_components=fixed_components,
				tolerance=tolerance,
			)
			if df.empty:
				raise
		for col in [comp_a, comp_b, comp_c]:
			if col not in df.columns:
				raise ValueError(f"Ternary component not found: {col}")

		s = df[comp_a].to_numpy(dtype=float) + df[comp_b].to_numpy(dtype=float) + df[comp_c].to_numpy(dtype=float)
		valid = s > 1e-12
		df = df.loc[valid].copy()
		s = s[valid]
		if df.empty:
			raise ValueError(f"No valid ternary rows for {comp_a}-{comp_b}-{comp_c} after filtering.")

		if phase_extrema_filter:
			df = self._reduce_slice_by_phase_extrema(
				df=df,
				composition_cols=[comp_a, comp_b, comp_c],
				tol=tolerance,
			)
			if df.empty:
				raise ValueError("Phase-extrema filter removed all rows for this ternary slice.")
			s = df[comp_a].to_numpy(dtype=float) + df[comp_b].to_numpy(dtype=float) + df[comp_c].to_numpy(dtype=float)
		else:
			# Mesh surfaces require per-composition extrema filtering for stable triangulation.
			ternary_phase_mesh = False

		df["a_norm"] = df[comp_a].to_numpy(dtype=float) / s
		df["b_norm"] = df[comp_b].to_numpy(dtype=float) / s
		df["c_norm"] = df[comp_c].to_numpy(dtype=float) / s
		df["ternary_slice_total"] = s
		if "is_interpolated" not in df.columns:
			df["is_interpolated"] = False
		tx, ty = self._cartesian_to_ternary_display(
			df["a_norm"].to_numpy(dtype=float),
			df["b_norm"].to_numpy(dtype=float),
		)
		df["tx"] = tx
		df["ty"] = ty

		fig = go.Figure()
		if color_by == "Phase":
			color_map = self._phase_color_map(df["Phase"].astype(str).tolist())

			def _add_ternary_scatter(g: pd.DataFrame, phase_name: str, showlegend: bool) -> None:
				fig.add_trace(
					go.Scatter3d(
						x=g["tx"],
						y=g["ty"],
						z=g["T_K"],
						mode="markers",
						name=str(phase_name),
						showlegend=showlegend,
						marker=dict(size=4, color=color_map[str(phase_name)], opacity=0.8),
						customdata=np.column_stack([
							g[comp_a].to_numpy(dtype=float),
							g[comp_b].to_numpy(dtype=float),
							g[comp_c].to_numpy(dtype=float),
							g["ternary_slice_total"].to_numpy(dtype=float),
							g["is_interpolated"].astype(str).to_numpy(),
							g["coexisting_phases"].astype(str).to_numpy(),
						]),
						hovertemplate=(
							f"Phase={phase_name}<br>"
							"T=%{z:.2f} K<br>"
							f"{comp_a}=%{{customdata[0]:.4f}}<br>"
							f"{comp_b}=%{{customdata[1]:.4f}}<br>"
							f"{comp_c}=%{{customdata[2]:.4f}}<br>"
							"Slice_total=%{customdata[3]:.4f}<br>"
							"Interpolated=%{customdata[4]}<br>"
							"Coexisting=%{customdata[5]}<extra></extra>"
						),
					)
				)

			def _add_ternary_mesh(g: pd.DataFrame, phase_name: str, showlegend: bool) -> bool:
				if len(g) < 3:
					return False
				xy = np.column_stack([
					g["tx"].to_numpy(dtype=float),
					g["ty"].to_numpy(dtype=float),
				])
				_, uniq_idx = np.unique(np.round(xy, 10), axis=0, return_index=True)
				if len(uniq_idx) < 3:
					return False
				uniq_idx = np.sort(uniq_idx)
				g_u = g.iloc[uniq_idx].copy().reset_index(drop=True)
				try:
					xy_u = np.column_stack([
						g_u["tx"].to_numpy(dtype=float),
						g_u["ty"].to_numpy(dtype=float),
					])
					triangles = Delaunay(xy_u).simplices
				except QhullError:
					return False
				if len(triangles) == 0:
					return False

				fig.add_trace(
					go.Mesh3d(
						x=g_u["tx"].to_numpy(dtype=float),
						y=g_u["ty"].to_numpy(dtype=float),
						z=g_u["T_K"].to_numpy(dtype=float),
						i=triangles[:, 0],
						j=triangles[:, 1],
						k=triangles[:, 2],
						name=str(phase_name),
						showlegend=showlegend,
						color=color_map[str(phase_name)],
						opacity=0.65,
						flatshading=True,
						customdata=np.column_stack([
							g_u[comp_a].to_numpy(dtype=float),
							g_u[comp_b].to_numpy(dtype=float),
							g_u[comp_c].to_numpy(dtype=float),
							g_u["ternary_slice_total"].to_numpy(dtype=float),
							g_u["is_interpolated"].astype(str).to_numpy(),
							g_u["coexisting_phases"].astype(str).to_numpy(),
						]),
						hovertemplate=(
							f"Phase={phase_name}<br>"
							"T=%{z:.2f} K<br>"
							f"{comp_a}=%{{customdata[0]:.4f}}<br>"
							f"{comp_b}=%{{customdata[1]:.4f}}<br>"
							f"{comp_c}=%{{customdata[2]:.4f}}<br>"
							"Slice_total=%{customdata[3]:.4f}<br>"
							"Interpolated=%{customdata[4]}<br>"
							"Coexisting=%{customdata[5]}<extra></extra>"
						),
					)
				)
				return True

			mesh_enabled = bool(ternary_phase_mesh)
			for phase_name, g in df.groupby("Phase"):
				phase_str = str(phase_name)
				showlegend = True
				is_liq = _phase_is_liquid(phase_str, self.liquid_aliases)
				is_ss = self._phase_is_solution(phase_str)

				if mesh_enabled and (is_liq or is_ss):
					if is_liq:
						if not _add_ternary_mesh(g, phase_str, showlegend=showlegend):
							_add_ternary_scatter(g, phase_str, showlegend=showlegend)
						continue

					cluster_base = float(slice_grid_delta) if (slice_grid_delta is not None and float(slice_grid_delta) > 0.0) else self._resolve_tol(tolerance)
					cluster_tol = max(cluster_base * float(max(ss_cluster_factor, 1.0)), self._resolve_tol(tolerance))
					mesh_points = g[["tx", "ty"]].to_numpy(dtype=float)
					components = _connected_components(mesh_points, threshold=cluster_tol + 1e-12)
					for comp_i, comp_idx in enumerate(components):
						g_cluster = g.iloc[comp_idx].copy()
						mesh_ok = _add_ternary_mesh(g_cluster, phase_str, showlegend=showlegend and comp_i == 0)
						if not mesh_ok:
							_add_ternary_scatter(g_cluster, phase_str, showlegend=showlegend and comp_i == 0)
					continue

				_add_ternary_scatter(g, phase_str, showlegend=showlegend)
		else:
			fig.add_trace(
				go.Scatter3d(
					x=df["tx"],
					y=df["ty"],
					z=df["T_K"],
					mode="markers",
					name=color_by,
					marker=dict(
						size=4,
						color=df[color_by].to_numpy(dtype=float),
						colorscale="Viridis",
						showscale=True,
						opacity=0.8,
					),
				)
			)

		# Draw base simplex triangle at minimum temperature for orientation.
		tmin = float(df["T_K"].min()) if not df.empty else 0.0
		b0x, b0y = self._cartesian_to_ternary_display(np.array([0.0]), np.array([0.0]))
		b1x, b1y = self._cartesian_to_ternary_display(np.array([1.0]), np.array([0.0]))
		b2x, b2y = self._cartesian_to_ternary_display(np.array([0.0]), np.array([1.0]))
		fig.add_trace(
			go.Scatter3d(
				x=[float(b0x[0]), float(b1x[0]), float(b2x[0]), float(b0x[0])],
				y=[float(b0y[0]), float(b1y[0]), float(b2y[0]), float(b0y[0])],
				z=[tmin, tmin, tmin, tmin],
				mode="lines",
				line=dict(color="black", width=3),
				name="slice base",
				showlegend=False,
			)
		)

		# Label the three simplex corners for the active ternary slice components.
		fig.add_trace(
			go.Scatter3d(
				x=[float(b0x[0]), float(b1x[0]), float(b2x[0])],
				y=[float(b0y[0]), float(b1y[0]), float(b2y[0])],
				z=[tmin, tmin, tmin],
				mode="text",
				text=[str(comp_c), str(comp_a), str(comp_b)],
				textposition="top center",
				showlegend=False,
				hoverinfo="skip",
			)
		)

		fig.update_layout(
			title=title or f"Ternary Slice 3D ({comp_a}-{comp_b}-{comp_c})",
			legend=dict(x=0.95, y=0.95, xanchor='left', yanchor='top'),
			autosize=False,
			width=1050,
			height=760,
			margin=dict(l=50, r=50, b=50, t=50),
			scene=dict(
				xaxis=dict(title="Ternary display x"),
				yaxis=dict(title="Ternary display y"),
				zaxis=dict(title="T [K]", range=list(temp_axis_range_k) if temp_axis_range_k is not None else None),
				bgcolor='white',
				camera=dict(projection=dict(type='orthographic')),
			),
		)
		return fig

	def plot_quaternary_phase_tetrahedral(
		self,
		phase_filter: str,
		temperature_extrema: Optional[str] = None,
		composition_tol: Optional[float] = None,
		title: Optional[str] = None,
		marker_size: float = 5.0,
		colorscale: str = "Viridis",
	) -> go.Figure:
		self._ensure_quaternary()
		df = self._df_full.copy()
		filter_token = str(phase_filter).strip()
		filter_up = filter_token.upper()

		if filter_up in {"L", "LIQUID"}:
			df = df[df["Phase"].apply(lambda p: _phase_is_liquid(p, self.liquid_aliases))].copy()
		elif "*" in filter_token or "?" in filter_token:
			pat = filter_token.upper()
			df = df[df["Phase"].astype(str).apply(lambda p: fnmatch.fnmatch(str(p).upper(), pat))].copy()
		else:
			df = df[df["Phase"].astype(str).str.upper() == filter_up].copy()

		if df.empty:
			raise ValueError(f"No rows available for phase filter: {phase_filter}")

		temp_col_name = "T_K"
		if temperature_extrema is not None:
			ext = str(temperature_extrema).strip().lower()
			if ext not in {"min", "max"}:
				raise ValueError("temperature_extrema must be one of: None, 'min', 'max'")

			tol = self._resolve_tol(composition_tol)
			df = df.copy()
			df["_qkey"] = self._quantize_composition_for_grouping(df, tol)
			idx = df.groupby("_qkey")["T_K"].idxmin() if ext == "min" else df.groupby("_qkey")["T_K"].idxmax()
			df = df.loc[idx].copy().reset_index(drop=True)
			temp_col_name = "T_min_K" if ext == "min" else "T_max_K"
			df[temp_col_name] = df["T_K"].astype(float)

		df = self._attach_coexisting_phases(df)
		if "is_interpolated" not in df.columns:
			df["is_interpolated"] = False
		arr4 = df[self._full_comp_cols].to_numpy(dtype=float)
		xyz = self._barycentric_to_tetrahedral(arr4)
		temp_vals = df["T_K"].to_numpy(dtype=float)
		hover = np.column_stack([
			df["Phase"].astype(str).to_numpy(),
			temp_vals,
			arr4,
			df["is_interpolated"].astype(str).to_numpy(),
			df["coexisting_phases"].astype(str).to_numpy(),
		])

		fig = go.Figure()
		fig.add_trace(
			go.Scatter3d(
				x=xyz[:, 0],
				y=xyz[:, 1],
				z=xyz[:, 2],
				mode="markers",
				marker=dict(
					size=float(marker_size),
					color=temp_vals,
					colorscale=colorscale,
					showscale=True,
					colorbar=dict(title=f"{temp_col_name.replace('_', ' ')}", x=0.93, len=0.78, thickness=16),
					opacity=0.85,
				),
				customdata=hover,
				hovertemplate=(
					"Phase=%{customdata[0]}<br>"
					f"{temp_col_name}=%{{customdata[1]:.2f}} K<br>"
					"x_dep=%{customdata[2]:.4f}<br>"
					"x0=%{customdata[3]:.4f}<br>"
					"x1=%{customdata[4]:.4f}<br>"
					"x2=%{customdata[5]:.4f}<br>"
					"Interpolated=%{customdata[6]}<br>"
					"Coexisting=%{customdata[7]}<extra></extra>"
				),
				name=f"{phase_filter} points",
			)
		)
		self._add_tetrahedron_wireframe(fig)

		labels = [self._component_label(i) for i in range(4)]
		fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="text", text=[labels[0]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[1.0], y=[0.0], z=[0.0], mode="text", text=[labels[1]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[0.5], y=[np.sqrt(3.0) / 2.0], z=[0.0], mode="text", text=[labels[2]], showlegend=False))
		fig.add_trace(
			go.Scatter3d(
				x=[0.5], y=[np.sqrt(3.0) / 6.0], z=[np.sqrt(2.0 / 3.0)], mode="text", text=[labels[3]], showlegend=False
			)
		)

		fig.update_layout(
			title=title or f"Quaternary {phase_filter} Temperature Map",
			autosize=False,
			width=1050,
			height=760,
			legend=dict(x=0.95, y=0.95, xanchor='left', yanchor='top'),
			scene=dict(
				xaxis=dict(visible=False),
				yaxis=dict(visible=False),
				zaxis=dict(visible=False),
				aspectmode="data",
				bgcolor='white',
				camera=dict(projection=dict(type='orthographic')),
			),
			margin=dict(l=50, r=50, b=50, t=50),
		)
		return fig

	def extract_intermetallic_single_composition_maxima(
		self,
		composition_tol: Optional[float] = None,
		include_solution_phases: bool = False,
	) -> pd.DataFrame:
		"""Return one row per single-composition phase at its highest temperature."""
		self._ensure_quaternary()
		df = self._df_full.copy()

		# Intermetallic view excludes liquid and, by default, excludes solution phases.
		non_liquid_mask = ~df["Phase"].apply(lambda p: _phase_is_liquid(p, self.liquid_aliases))
		df = df[non_liquid_mask].copy()
		if not include_solution_phases:
			df = df[~df["Phase"].apply(self._phase_is_solution)].copy()

		if df.empty:
			return df

		tol = self._resolve_tol(composition_tol)
		qkey = self._quantize_composition_for_grouping(df, tol)
		df["_qkey"] = qkey

		phase_unique_comp = df.groupby("Phase")["_qkey"].nunique()
		single_comp_phases = phase_unique_comp[phase_unique_comp == 1].index
		df = df[df["Phase"].isin(single_comp_phases)].copy()
		if df.empty:
			return df

		idx = df.groupby("Phase")["T_K"].idxmax()
		out = df.loc[idx].copy().sort_values(by=["T_K", "Phase"], ascending=[False, True]).reset_index(drop=True)
		return out.drop(columns=["_qkey"], errors="ignore")

	@staticmethod
	def _barycentric_to_tetrahedral(arr4: np.ndarray) -> np.ndarray:
		v = np.array([
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.5, np.sqrt(3.0) / 2.0, 0.0],
			[0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
		], dtype=float)
		return arr4 @ v

	@staticmethod
	def _add_tetrahedron_wireframe(fig: go.Figure):
		verts = np.array([
			[0.0, 0.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.5, np.sqrt(3.0) / 2.0, 0.0],
			[0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
		], dtype=float)
		edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
		for i, j in edges:
			fig.add_trace(
				go.Scatter3d(
					x=[verts[i, 0], verts[j, 0]],
					y=[verts[i, 1], verts[j, 1]],
					z=[verts[i, 2], verts[j, 2]],
					mode="lines",
					line=dict(color="rgba(80,80,80,0.7)", width=2),
					showlegend=False,
					hoverinfo="skip",
				)
			)

	def _build_quaternary_point_frame(
		self,
		solution_only: bool,
		include_liquid: bool,
	) -> pd.DataFrame:
		self._ensure_quaternary()
		df = self._df_full.copy()
		if not include_liquid:
			df = df[~df["Phase"].apply(lambda p: _phase_is_liquid(p, self.liquid_aliases))].copy()
		if solution_only:
			df = df[df["Phase"].apply(self._phase_is_solution)].copy()
		return df.reset_index(drop=True)

	def plot_quaternary_tetrahedral(
		self,
		solution_only: bool = True,
		include_liquid: bool = False,
		title: Optional[str] = None,
		marker_size: float = 5.0,
	) -> go.Figure:
		"""Plot quaternary boundary points on a tetrahedron colored by temperature."""
		df = self._build_quaternary_point_frame(solution_only=solution_only, include_liquid=include_liquid)
		if df.empty:
			raise ValueError("No rows available for quaternary tetrahedral solution map under current filters.")

		arr4 = df[self._full_comp_cols].to_numpy(dtype=float)
		xyz = self._barycentric_to_tetrahedral(arr4)

		phase_vals = df["Phase"].astype(str).to_numpy()
		df = self._attach_coexisting_phases(df)
		hover = np.column_stack([
			phase_vals,
			df["T_K"].to_numpy(dtype=float),
			arr4,
			df["coexisting_phases"].astype(str).to_numpy(),
		])

		fig = go.Figure()
		fig.add_trace(
			go.Scatter3d(
				x=xyz[:, 0],
				y=xyz[:, 1],
				z=xyz[:, 2],
				mode="markers",
				marker=dict(
					size=float(marker_size),
					color=df["T_K"].to_numpy(dtype=float),
					colorscale="Viridis",
					showscale=True,
					colorbar=dict(title="T [K]", x=0.93, len=0.78, thickness=16),
					opacity=0.85,
				),
				customdata=hover,
				hovertemplate=(
					"Phase=%{customdata[0]}<br>"
					"T=%{customdata[1]:.2f} K<br>"
					"x_dep=%{customdata[2]:.4f}<br>"
					"x0=%{customdata[3]:.4f}<br>"
					"x1=%{customdata[4]:.4f}<br>"
					"x2=%{customdata[5]:.4f}<br>"
					"Coexisting=%{customdata[6]}<extra></extra>"
				),
				name="Boundary points",
			)
		)
		self._add_tetrahedron_wireframe(fig)

		labels = [self._component_label(i) for i in range(4)]
		fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="text", text=[labels[0]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[1.0], y=[0.0], z=[0.0], mode="text", text=[labels[1]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[0.5], y=[np.sqrt(3.0) / 2.0], z=[0.0], mode="text", text=[labels[2]], showlegend=False))
		fig.add_trace(
			go.Scatter3d(
				x=[0.5], y=[np.sqrt(3.0) / 6.0], z=[np.sqrt(2.0 / 3.0)], mode="text", text=[labels[3]], showlegend=False
			)
		)

		fig.update_layout(
			title=title or "Quaternary Tetrahedral Boundary Temperature Map",
			autosize=False,
			width=1050,
			height=760,
			legend=dict(x=0.95, y=0.95, xanchor='left', yanchor='top'),
			scene=dict(
				xaxis=dict(visible=False),
				yaxis=dict(visible=False),
				zaxis=dict(visible=False),
				aspectmode="data",
				bgcolor='white',
				camera=dict(projection=dict(type='orthographic')),
			),
			margin=dict(l=50, r=50, b=50, t=50),
		)
		return fig

	def plot_quaternary_intermetallic_max_t(
		self,
		composition_tol: Optional[float] = None,
		include_solution_phases: bool = False,
		title: Optional[str] = None,
		marker_size: float = 8.0,
	) -> go.Figure:
		"""Plot intermetallic single-composition phase maxima on one tetrahedron."""
		max_df = self.extract_intermetallic_single_composition_maxima(
			composition_tol=composition_tol,
			include_solution_phases=include_solution_phases,
		)
		if max_df.empty:
			raise ValueError(
				"No intermetallic single-composition phases found for maxima plot under current filters."
			)

		arr4 = max_df[self._full_comp_cols].to_numpy(dtype=float)
		xyz = self._barycentric_to_tetrahedral(arr4)
		max_df = self._attach_coexisting_phases(max_df)
		hover = np.column_stack([
			max_df["Phase"].astype(str).to_numpy(),
			max_df["T_K"].to_numpy(dtype=float),
			arr4,
			max_df["coexisting_phases"].astype(str).to_numpy(),
		])

		fig = go.Figure()
		fig.add_trace(
			go.Scatter3d(
				x=xyz[:, 0],
				y=xyz[:, 1],
				z=xyz[:, 2],
				mode="markers+text",
				text=max_df["Phase"].astype(str).tolist(),
				textposition="top center",
				marker=dict(
					size=float(marker_size),
					color=max_df["T_K"].to_numpy(dtype=float),
					colorscale="Plasma",
					showscale=True,
					colorbar=dict(title="T_max [K]", x=0.93, len=0.78, thickness=16),
					opacity=0.95,
					line=dict(color="rgba(0,0,0,0.45)", width=1),
				),
				customdata=hover,
				hovertemplate=(
					"Phase=%{customdata[0]}<br>"
					"T_max=%{customdata[1]:.2f} K<br>"
					"x_dep=%{customdata[2]:.4f}<br>"
					"x0=%{customdata[3]:.4f}<br>"
					"x1=%{customdata[4]:.4f}<br>"
					"x2=%{customdata[5]:.4f}<br>"
					"Coexisting=%{customdata[6]}<extra></extra>"
				),
				name="Intermetallic phase maxima",
			)
		)
		self._add_tetrahedron_wireframe(fig)

		labels = [self._component_label(i) for i in range(4)]
		fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="text", text=[labels[0]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[1.0], y=[0.0], z=[0.0], mode="text", text=[labels[1]], showlegend=False))
		fig.add_trace(go.Scatter3d(x=[0.5], y=[np.sqrt(3.0) / 2.0], z=[0.0], mode="text", text=[labels[2]], showlegend=False))
		fig.add_trace(
			go.Scatter3d(
				x=[0.5], y=[np.sqrt(3.0) / 6.0], z=[np.sqrt(2.0 / 3.0)], mode="text", text=[labels[3]], showlegend=False
			)
		)

		fig.update_layout(
			title=title or "Quaternary Intermetallic Single-Composition Phase Maxima",
			autosize=False,
			width=1050,
			height=760,
			legend=dict(x=0.95, y=0.95, xanchor='left', yanchor='top'),
			scene=dict(
				xaxis=dict(visible=False),
				yaxis=dict(visible=False),
				zaxis=dict(visible=False),
				aspectmode="data",
				bgcolor='white',
				camera=dict(projection=dict(type='orthographic')),
			),
			margin=dict(l=50, r=50, b=50, t=50),
		)
		return fig


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


def _parse_csv_env(name: str) -> Optional[List[str]]:
	raw = os.getenv(name)
	if raw is None:
		return None
	parts = [p.strip() for p in raw.split(",") if p.strip()]
	return parts if parts else None


def main_post() -> None:
	default_dir = "all_dumps/quaternary_demo/lower_hull_cache/Hf-Nb-W-Zr"
	system_dir = os.getenv("GP_SYSTEM_CACHE_DIR", default_dir)
	recursive = _parse_bool_env("GP_RECURSIVE", True)
	load_gtx = _parse_bool_env("GP_LOAD_GTX", True)
	load_and_stitch = _parse_bool_env("GP_LOAD_AND_STITCH", True)
	extract_phase_boundary = _parse_bool_env("GP_EXTRACT_PHASE_BOUNDARY", True)
	save_phase_boundary = _parse_bool_env("GP_SAVE_PHASE_BOUNDARY", True)
	dump_full_csv = _parse_bool_env("GP_DUMP_FULL_EQUILIBRIUM_CSV", False)
	sample_max_rows = _parse_int_env("GP_SAMPLE_MAX_ROWS", 10000)
	notify_tmax_unchanged = _parse_bool_env("GP_NOTIFY_TMAX_UNCHANGED", True)
	notify_tmin_contains_liquid = _parse_bool_env("GP_NOTIFY_TMIN_CONTAINS_LIQUID", True)
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
	print(f"Dump full equilibrium CSV: {dump_full_csv}")
	print(f"Sample max rows (when not full): {sample_max_rows}")
	print(f"Notify when Tmax unchanged after filter: {notify_tmax_unchanged}")
	print(f"Notify when Tmin contains liquid: {notify_tmin_contains_liquid}")
	print(f"Tmax change tolerance: {tmax_change_atol}")
	print(f"Plotter dir override: {plotter_dir_override}")
	print(f"Rows to print from reconstructed equilibrium_df: {print_rows}")

	post = GeneralPostProcess(
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
				sample_csv_path = post.save_phase_boundary_sample_csv(
					output_dir=plotter_dir_override,
					max_rows=sample_max_rows,
					dump_full=dump_full_csv,
				)
				if dump_full_csv:
					print(f"Saved phase-boundary full CSV: {sample_csv_path}")
				else:
					print(f"Saved phase-boundary sample CSV: {sample_csv_path}")

			tmax_diag = post.evaluate_tmax_filter_change(atol=tmax_change_atol)
			print("\n[Temperature-boundary diagnostic]")
			print(
				f"  full_tmax_k={tmax_diag['full_tmax_k']} | "
				f"filtered_tmax_k={tmax_diag['filtered_tmax_k']} | "
				f"tmax_changed_after_filter={tmax_diag['tmax_changed_after_filter']}"
			)
			print(f"  all_phases_at_full_tmax_are_liquid={tmax_diag['all_phases_at_full_tmax_are_liquid']}")
			print(f"  phases_at_full_tmax={tmax_diag['phases_at_full_tmax']}")
			print(f"  full_tmin_k={tmax_diag['full_tmin_k']}")
			print(f"  liquid_present_at_full_tmin={tmax_diag['liquid_present_at_full_tmin']}")
			print(f"  phases_at_full_tmin={tmax_diag['phases_at_full_tmin']}")

			if notify_tmax_unchanged and (tmax_diag.get("tmax_changed_after_filter") is False):
				print(
					"[WARN] Boundary-filter Tmax did not decrease. "
					"Highest-temperature rows still include non-single-phase boundary simplices; "
					"consider sampling a higher Tmax range to ensure complete high-T closure."
				)

			if notify_tmin_contains_liquid and (tmax_diag.get("liquid_present_at_full_tmin") is True):
				print(
					"[WARN] Lowest-temperature slice still contains liquid phase rows. "
					"Consider extending the temperature range lower to capture full low-T closure."
				)

		s = post.summary()
		print("\n[Summary]")
		print(json.dumps(s, indent=2, default=str))
	else:
		print("\n[Stitch] Skipped. Set GP_LOAD_AND_STITCH=true to run full stitching.")

	print("=" * 72)


def main_viz() -> None:
	default_phase_boundary_file = (
		"all_dumps/quaternary_demo/plotter_dir/"
		"Hf-Nb-W-Zr_T1927.00_3677.00_phase_boundary_equilibrium.pkl.gz"
	)
	phase_boundary_file = os.getenv("GP_PHASE_BOUNDARY_FILE", default_phase_boundary_file)
	element_names = _parse_csv_env("GP_ELEMENT_NAMES")
	solution_only = _parse_bool_env("GP_VIZ_SOLUTION_ONLY", True)
	include_liquid = _parse_bool_env("GP_VIZ_INCLUDE_LIQUID", False)
	intermetallic_include_solution = _parse_bool_env("GP_VIZ_INTERMETALLIC_INCLUDE_SOLUTION", False)
	composition_tol = _parse_float_env("GP_VIZ_INTERMETALLIC_TOL", 1e-6)
	save_html = _parse_bool_env("GP_VIZ_SAVE_HTML", True)
	html_out_dir = Path(os.getenv("GP_VIZ_HTML_OUT_DIR", "all_dumps/quaternary_demo/plotter_dir"))
	grid_delta = _parse_float_env("GP_VIZ_GRID_DELTA", 0.025)
	slice_phase_extrema_filter = _parse_bool_env("GP_VIZ_SLICE_PHASE_EXTREMA_FILTER", True)
	ternary_phase_mesh = _parse_bool_env("GP_VIZ_TERNARY_PHASE_MESH", True)
	ss_cluster_factor = _parse_float_env("GP_VIZ_TERNARY_SS_CLUSTER_FACTOR", 1.75)
	if not slice_phase_extrema_filter:
		ternary_phase_mesh = False

	print("=" * 72)
	print("GENERAL PLOTTER VIZ SMOKE TEST")
	print("=" * 72)
	print(f"Phase boundary file: {phase_boundary_file}")
	print(f"Element names override: {element_names}")
	print(f"Solution-only quaternary map: {solution_only}")
	print(f"Include liquid in quaternary map: {include_liquid}")
	print(f"Intermetallic include solution phases: {intermetallic_include_solution}")
	print(f"Intermetallic composition tolerance: {composition_tol}")
	print(f"Save HTML outputs: {save_html}")
	print(f"HTML output dir: {html_out_dir}")
	print(f"Slice grid delta: {grid_delta}")
	print(f"Slice phase-extrema filter: {slice_phase_extrema_filter}")
	print(f"Ternary phase mesh enabled: {ternary_phase_mesh}")
	print(f"Ternary SS cluster factor: {ss_cluster_factor}")

	phase_boundary_path = Path(phase_boundary_file)
	if not phase_boundary_path.exists():
		raise FileNotFoundError(f"Phase-boundary file not found: {phase_boundary_path}")

	df = pd.read_pickle(phase_boundary_path, compression="gzip")
	comp_cols = _infer_comp_cols(df)
	if element_names is None:
		stem = phase_boundary_path.name
		if "_T" in stem:
			sys_name = stem.split("_T", 1)[0]
			element_names = [p.strip() for p in sys_name.split("-") if p.strip()]
	print(f"Loaded boundary dataframe shape: {df.shape}")
	print(f"Inferred composition columns: {comp_cols}")
	print(f"Element names used: {element_names}")

	plotter = PhaseBoundaryPlotter(
		equilibrium_df=df,
		composition_cols=comp_cols,
		element_names=element_names,
	)

	comp_check = plotter.validate_composition_integrity(tolerance=1e-6)
	print(f"Composition integrity: {comp_check}")
	print(f"Temperature stats: {plotter.get_temperature_statistics()}")

	# Fixed quaternary component names for this system after inference.
	full_names = plotter._full_component_names()
	if len(full_names) != 4:
		raise ValueError(f"Expected quaternary system (4 components), got {len(full_names)}: {full_names}")

	if set(["Hf", "Nb", "W", "Zr"]).issubset(set(full_names)):
		n_hf, n_nb, n_w, n_zr = "Hf", "Nb", "W", "Zr"
	else:
		# Fallback to positional mapping if names are unavailable or custom.
		n_hf, n_nb, n_w, n_zr = full_names[0], full_names[1], full_names[2], full_names[3]

	# Snap requested non-zero constraints to the available composition grid.
	full_df = plotter._full_component_df()
	nonzero_hf_target = plotter._snap_to_grid(0.05, grid_delta)
	nonzero_w_target = plotter._snap_to_grid(0.075, grid_delta)
	nonzero_w_tern_target = plotter._snap_to_grid(0.10, grid_delta)

	nonzero_hf = plotter._nearest_available_value(full_df[n_hf], nonzero_hf_target)
	nonzero_w = plotter._nearest_available_value(full_df[n_w], nonzero_w_target)
	nonzero_w_tern = plotter._nearest_available_value(full_df[n_w], nonzero_w_tern_target)

	print(f"Chosen non-zero binary slice fixed values: {n_hf}={nonzero_hf}, {n_w}={nonzero_w}")
	print(f"Chosen non-zero ternary slice fixed value: {n_w}={nonzero_w_tern}")

	# 1) Binary Nb-Zr at Hf=0, W=0.
	fig_bin_pure = plotter.plot_binary_slice_tx(
		comp_a=n_nb,
		comp_b=n_zr,
		fixed_components={n_hf: 0.0, n_w: 0.0},
		tolerance=max(0.5 * grid_delta, 1e-6),
		phase_extrema_filter=slice_phase_extrema_filter,
		title=f"{n_nb}-{n_zr} Binary Slice ({n_hf}=0, {n_w}=0)",
	)

	# 2) Binary Nb-Zr at non-zero Hf and W.
	fig_bin_nonzero = plotter.plot_binary_slice_tx(
		comp_a=n_nb,
		comp_b=n_zr,
		fixed_components={n_hf: nonzero_hf, n_w: nonzero_w},
		tolerance=max(0.5 * grid_delta, 1e-6),
		phase_extrema_filter=slice_phase_extrema_filter,
		title=f"{n_nb}-{n_zr} Slice ({n_hf}={nonzero_hf:.3f}, {n_w}={nonzero_w:.3f})",
	)

	# 3) Ternary Hf-Nb-Zr at W=0.
	fig_tern_w0 = plotter.plot_ternary_slice(
		comp_a=n_hf,
		comp_b=n_nb,
		comp_c=n_zr,
		fixed_components={n_w: 0.0},
		tolerance=max(0.5 * grid_delta, 1e-6),
		phase_extrema_filter=slice_phase_extrema_filter,
		ternary_phase_mesh=ternary_phase_mesh,
		slice_grid_delta=grid_delta,
		ss_cluster_factor=ss_cluster_factor,
		title=f"{n_hf}-{n_nb}-{n_zr} Slice ({n_w}=0)",
		color_by="Phase",
	)

	# 4) Ternary Hf-Nb-Zr at non-zero W.
	fig_tern_wnz = plotter.plot_ternary_slice(
		comp_a=n_hf,
		comp_b=n_nb,
		comp_c=n_zr,
		fixed_components={n_w: nonzero_w_tern},
		tolerance=max(0.5 * grid_delta, 1e-6),
		phase_extrema_filter=slice_phase_extrema_filter,
		ternary_phase_mesh=ternary_phase_mesh,
		slice_grid_delta=grid_delta,
		ss_cluster_factor=ss_cluster_factor,
		title=f"{n_hf}-{n_nb}-{n_zr} Slice ({n_w}={nonzero_w_tern:.3f})",
		color_by="Phase",
	)

	fig_liquid = plotter.plot_quaternary_phase_tetrahedral(
		phase_filter="L",
		temperature_extrema="min",
		composition_tol=max(0.5 * grid_delta, 1e-6),
		title="Quaternary Liquidus Minimum-Temperature Map",
	)
	print(f"Liquid tetrahedral traces: {len(fig_liquid.data)}")

	phase_list = plotter.get_phase_list(include_liquid=True)
	bcc_candidates = [p for p in phase_list if "BCC" in str(p).upper()]
	if not bcc_candidates:
		raise ValueError(f"No BCC-like phase found in phase list: {phase_list}")
	bcc_phase_name = sorted(bcc_candidates)[0]

	fig_bcc = plotter.plot_quaternary_phase_tetrahedral(
		phase_filter="BCC*",
		temperature_extrema="max",
		composition_tol=max(0.5 * grid_delta, 1e-6),
		title=f"Quaternary {bcc_phase_name} Maximum-Temperature Map",
	)
	print(f"{bcc_phase_name} tetrahedral traces: {len(fig_bcc.data)}")

	fig_intermetallic = plotter.plot_quaternary_intermetallic_max_t(
		composition_tol=composition_tol,
		include_solution_phases=intermetallic_include_solution,
		title="Quaternary Intermetallic Max Temperature Map",
	)
	print(f"Intermetallic tetrahedral traces: {len(fig_intermetallic.data)}")

	if save_html:
		html_out_dir.mkdir(parents=True, exist_ok=True)
		bin_pure_out = html_out_dir / "nb_zr_binary_hf0_w0.html"
		bin_nonzero_out = html_out_dir / "nb_zr_binary_nonzero_hf_w.html"
		tern_w0_out = html_out_dir / "hf_nb_zr_ternary_w0.html"
		tern_wnz_out = html_out_dir / "hf_nb_zr_ternary_w_nonzero.html"
		liq_out = html_out_dir / "quaternary_liquid_min_t_map.html"
		bcc_out = html_out_dir / "quaternary_bcc_max_t_map.html"
		imt_out = html_out_dir / "quaternary_intermetallic_max_t_map.html"
		fig_bin_pure.write_html(str(bin_pure_out), include_plotlyjs="cdn")
		fig_bin_nonzero.write_html(str(bin_nonzero_out), include_plotlyjs="cdn")
		fig_tern_w0.write_html(str(tern_w0_out), include_plotlyjs="cdn")
		fig_tern_wnz.write_html(str(tern_wnz_out), include_plotlyjs="cdn")
		fig_liquid.write_html(str(liq_out), include_plotlyjs="cdn")
		fig_bcc.write_html(str(bcc_out), include_plotlyjs="cdn")
		fig_intermetallic.write_html(str(imt_out), include_plotlyjs="cdn")
		print(f"Saved HTML: {bin_pure_out}")
		print(f"Saved HTML: {bin_nonzero_out}")
		print(f"Saved HTML: {tern_w0_out}")
		print(f"Saved HTML: {tern_wnz_out}")
		print(f"Saved HTML: {liq_out}")
		print(f"Saved HTML: {bcc_out}")
		print(f"Saved HTML: {imt_out}")

	print("=" * 72)


if __name__ == "__main__":
    # main_post()
    main_viz()

