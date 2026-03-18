from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import sympy as sp

import gliquid.config as cfg

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("C:\\Users\\AbrarRauf\\University of Michigan Dropbox\\Abrar Rauf\\WHSun_Lab\\G_liquid")
print(f"Setting workspace root to: {WORKSPACE_ROOT}")

cfg.set_project_root(WORKSPACE_ROOT)
cfg.set_data_dir(Path(cfg.project_root / "matrix_data"))
cfg.set_dir_structure(cfg._DIR_STRUCT_OPTS[1])

import gliquid.load_binary_data as lbd
from gliquid.binary import (
    BLPlotter,
    BinaryLiquid,
    _x_prec,
    _x_vals,
    a_sym,
    b_sym,
    c_sym,
    d_sym,
    t_sym,
    xb_sym,
)
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDPlotter, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None


EV_PER_ATOM_TO_J_PER_MOL = 96485.0
R = 8.314

ALL_SS_PHASES = ["BCC", "FCC", "HCP"]
SS_SYMBOLS = {
    "BCC": "Im-3m",
    "FCC": "Fm-3m",
    "HCP": "P6_3/mmc",
}
SS_SGS = {
    "BCC": 229,
    "FCC": 225,
    "HCP": 194,
}
DEFAULT_REF_MODE = "omegas-legacy"  # Options: "local-cache", "omegas-legacy"
DEFAULT_OMEGAS_PATH = cfg.data_dir / "omegas.json"

def _safe_spacegroup(structure) -> tuple[int | None, str | None]:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        return int(sga.get_space_group_number()), str(sga.get_space_group_symbol())
    except Exception:
        return None, None


def get_ss_model_refs_from_local_entries(
    components: list[str],
    dft_ch: PhaseDiagram
) -> dict[str, dict[str, float | str | int]]:
    
    entries = dft_ch.all_entries

    ss_models = {ss_phase: {} for ss_phase in ALL_SS_PHASES}

    ground_refs: dict[str, dict[str, float | str | int]] = {}
    phase_refs: dict[str, dict[str, dict[str, float | str | int]]] = {}

    for el in components:
        pure_entries = [entry for entry in entries if entry.composition.reduced_formula == el]
        if not pure_entries:
            raise RuntimeError(f"No pure-element entries found for '{el}'.")

        ground = min(pure_entries, key=lambda entry: float(entry.energy_per_atom))
        ground_sg, ground_symbol = _safe_spacegroup(getattr(ground, "structure", None))

        ground_refs[el] = {
            "source": "local-cache",
            "ground_material_id": str(getattr(ground, "entry_id", "unknown")),
            "ground_spacegroup": int(ground_sg) if ground_sg is not None else -1,
            "ground_symbol": ground_symbol or "unknown",
            "ground_energy_ev_per_atom": float(ground.energy_per_atom),
            "ss_phases": []
        }

        candidates: dict[str, list[ComputedEntry]] = {}
        for entry in pure_entries:
            sg_num, _ = _safe_spacegroup(getattr(entry, "structure", None))
            for phase, sg in SS_SGS.items():
                if sg_num == sg:
                    if phase not in candidates:
                        candidates[phase] = []
                    candidates[phase].append(entry)

        for ss_phase in ALL_SS_PHASES:
            if ss_phase not in candidates.keys():
                print(f"No {ss_phase} (spacegroup {SS_SGS[ss_phase]}) pure entry found for '{el}' in local cache.")
                continue
            e = min(candidates[ss_phase], key=lambda entry: float(entry.energy_per_atom))
            sg, symbol = _safe_spacegroup(getattr(e, "structure", None))
            delta_e_ev = float(e.energy_per_atom) - float(ground.energy_per_atom)
            phase_refs[ss_phase][el] = {
                "material_id": str(getattr(e, "entry_id", "unknown")),
                "spacegroup": int(sg) if sg is not None else SS_SGS[ss_phase],
                "symbol": symbol or SS_SYMBOLS[ss_phase],
                "energy_ev_per_atom": float(e.energy_per_atom),
                "delta_e_ev": float(delta_e_ev),
                "delta_e_jmol": float(delta_e_ev * EV_PER_ATOM_TO_J_PER_MOL)}

    for ss_phase in phase_refs.keys():
        refs = ground_refs.copy()
        for el in components:
            refs[el].update(phase_refs[ss_phase][el]) if el in phase_refs[ss_phase] else None
        ss_models[ss_phase].update({"refs": refs})

    print(ss_models)        
    return ss_models

def _build_single_ss_model_legacy(
    data: dict,
    components: list[str],
    ss_phase: str,
    pair_key: str,
) -> dict[str, float]:
    element_blocks: dict[str, dict[str, float]] = data["elements"]

    stable_pure = {}
    for el in components:
        candidates = [float(block[el]) for block in element_blocks.values() if el in block]
        if not candidates:
            raise KeyError(f"Could not find pure-element references for '{el}' in omegas file.")
        stable_pure[el] = min(candidates)

    refs = {}
    for el in components:
        ss_e = float(element_blocks[ss_phase][el])
        ground_e = stable_pure[el]
        delta_ev = ss_e - ground_e
        refs[el] = {
            "source": "omegas-legacy",
            "ground_material_id": "legacy",
            "ground_spacegroup": -1,
            "ground_symbol": "legacy",
            "ground_energy_ev_per_atom": float(ground_e),
            "material_id": "legacy",
            "spacegroup": SS_SGS.get(ss_phase, -1),
            "symbol": SS_SYMBOLS.get(ss_phase, "unknown"),
            "energy_ev_per_atom": float(ss_e),
            "delta_e_ev": float(delta_ev),
            "delta_e_jmol": float(delta_ev * EV_PER_ATOM_TO_J_PER_MOL),
        }

    offset_a_ev = float(refs[components[0]]["delta_e_ev"])
    offset_b_ev = float(refs[components[1]]["delta_e_ev"])
    omega_ev = float(data["omegas"][ss_phase][pair_key])

    return {
        "ref_mode": "omegas-legacy",
        "refs": refs,
        "omega_ev": omega_ev,
        "omega_jmol": omega_ev * EV_PER_ATOM_TO_J_PER_MOL,
        "offset_a_ev": offset_a_ev,
        "offset_b_ev": offset_b_ev,
        "offset_a_jmol": offset_a_ev * EV_PER_ATOM_TO_J_PER_MOL,
        "offset_b_jmol": offset_b_ev * EV_PER_ATOM_TO_J_PER_MOL,
    }


def load_solid_solution_models(
    omegas_path: Path,
    components: list[str],
    dft_ch: PhaseDiagram,
    ref_mode: str = DEFAULT_REF_MODE,
    ) -> dict[str, dict[str, float]]:

    data = json.loads(omegas_path.read_text(encoding="utf-8"))
    pair_key = "-".join(sorted(components))

    if ref_mode == "omegas-legacy":
        models: dict[str, dict[str, float]] = {}
        element_blocks: dict[str, dict[str, float]] = data.get("elements", {})
        omega_phases: dict[str, dict[str, float]] = data.get("omegas", {})

        for ss_phase, omega_block in omega_phases.items():
            if pair_key not in omega_block:
                continue
            if ss_phase not in element_blocks:
                continue
            if any(el not in element_blocks[ss_phase] for el in components):
                continue
            models[ss_phase] = _build_single_ss_model_legacy(
                data=data,
                components=components,
                ss_phase=ss_phase,
                pair_key=pair_key,
            )

        if not models:
            raise KeyError(
                f"No solid-solution phases found for pair '{pair_key}' with matching omegas/elements entries."
            )
        return models

    if ref_mode == "local-cache": # Need to update to allow for multiple SS phases

        ss_models = get_ss_model_refs_from_local_entries(components, dft_ch)
        for ss_phase, model_dict in ss_models.items():
            omega_block = data["omegas"].get(ss_phase, {})
            if pair_key not in omega_block:
                raise KeyError(f"Could not find omega for pair '{pair_key}' in phase '{ss_phase}'.")
            
            offset_a_ev = float(model_dict["refs"][components[0]]["delta_e_ev"])
            offset_b_ev = float(model_dict["refs"][components[1]]["delta_e_ev"])
            omega_ev = float(omega_block[pair_key])
            ss_models[ss_phase].update({
                "ref_mode": "local-cache",
                "omega_ev": omega_ev,
                "omega_jmol": omega_ev * EV_PER_ATOM_TO_J_PER_MOL,
                "offset_a_ev": offset_a_ev,
                "offset_b_ev": offset_b_ev,
                "offset_a_jmol": offset_a_ev * EV_PER_ATOM_TO_J_PER_MOL,
                "offset_b_jmol": offset_b_ev * EV_PER_ATOM_TO_J_PER_MOL,
            })

        return ss_models
    
    raise ValueError("ref_mode must be one of: local-cache, omegas-legacy")



class SolidSolutionBinaryLiquid(BinaryLiquid):
    """BinaryLiquid variant that injects a continuous solid-solution branch into HSX data."""

    def __init__(
        self,
        *args,
        ss_models: dict[str, dict[str, float]],
        **kwargs,
    ):
        self.ss_models = ss_models
        self.ss_names = list(ss_models.keys())

        super().__init__(*args, **kwargs)
        self._ensure_solid_solution_phase()

    def _ensure_solid_solution_phase(self) -> None:
        insert_idx = max(len(self.phases) - 1, 0)
        for ss_name in self.ss_names:
            if any(phase["name"] == ss_name for phase in self.phases):
                continue
            self.phases.insert(
                insert_idx,
                {"name": ss_name, "comp": 0.5, "enthalpy": 0.0, "entropy": 0.0, "points": []},
            )
            insert_idx += 1

    @classmethod
    def from_cache(cls, input, **kwargs) -> SolidSolutionBinaryLiquid:

        bl = BinaryLiquid.from_cache(input, **kwargs)

        ss_models = load_solid_solution_models(
            components=bl.components,
            dft_ch=bl.dft_ch,
            omegas_path=kwargs.get("omegas_path", DEFAULT_OMEGAS_PATH),
            ref_mode=kwargs.get("ref_mode", DEFAULT_REF_MODE),
        )
        
        kwargs.update({
            'sys_name': bl.sys_name,
            'components': bl.components,
            'component_data': bl.component_data,
            'mpds_json': bl.mpds_json,
            'digitized_liq': bl.digitized_liq,
            'temp_range': bl.temp_range,
            'dft_type': bl.dft_type,
            'dft_ch': bl.dft_ch,
            'phases': bl.phases,
            'params': bl._params,
            'param_format': bl._param_format,
            'eqs': bl.eqs,
            'comp_range_fit_lim': bl.comp_range_fit_lim,
            'init_error': bl.init_error
            })
        
        return cls(
            ss_models=ss_models,
            **kwargs
        )


    def solid_solution_h_s(self, ss_name: str, x_vals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        ss_model = self.ss_models[ss_name]
        x_arr = np.asarray(_x_vals if x_vals is None else x_vals, dtype=float)
        conf_term = np.zeros_like(x_arr)
        interior = (x_arr > 0.0) & (x_arr  < 1.0)
        xa = 1 - x_arr
        conf_term[interior] = xa[interior] * np.log(xa[interior]) + x_arr[interior] * np.log(x_arr[interior])
        s_conf = -R * conf_term

        h_offsets = ss_model["offset_a_jmol"] * xa + ss_model["offset_b_jmol"] * x_arr
        h_mix = ss_model["omega_jmol"] * xa * x_arr
        h_total = h_offsets + h_mix

        return h_total, s_conf

    def solid_solution_gibbs(self, ss_name: str, x_vals: np.ndarray, temp_k: float) -> np.ndarray:
        h_vals, s_vals = self.solid_solution_h_s(ss_name, x_vals=x_vals)
        return h_vals - float(temp_k) * s_vals

    def to_HSX(self, fmt: str = "dict"):  # noqa: N802
        lambda_args_vals = [
            _x_vals[1:-1],
            self.mean_elt_tm,
            self.get_L0_a(),
            self.get_L0_b(),
            self.get_L1_a(),
            self.get_L1_b(),
        ]
        liq_h_vals = self.eqs["h_liq_lambdified"](*lambda_args_vals).flatten().tolist()
        liq_s_vals = self.eqs["s_liq_lambdified"](*lambda_args_vals).flatten().tolist()

        lhs_h = self.component_data[self.components[0]]['H_liq']
        rhs_h = self.component_data[self.components[1]]['H_liq']
        lhs_s = self.component_data[self.components[0]]['S_liq'] if lhs_h != 0 else 0.0
        rhs_s = self.component_data[self.components[1]]['S_liq'] if rhs_h != 0 else 0.0

        data = {
            "X": [float(x) for x in _x_vals],
            "S": [float(lhs_s), *[float(s) for s in liq_s_vals], float(rhs_s)],
            "H": [float(lhs_h), *[float(h) for h in liq_h_vals], float(rhs_h)],
            "Phase Name": ["L"] * len(_x_vals),
        }

        for phase in self.phases:
            name = phase["name"]
            if name == "L" or name in self.ss_names:
                continue
            data["H"].append(float(phase["enthalpy"]))
            data["S"].append(float(phase.get("entropy", 0.0)))
            data["X"].append(round(float(phase["comp"]), _x_prec))
            data["Phase Name"].append(name)

        for ss_name in self.ss_names:
            ss_h_vals, ss_s_vals = self.solid_solution_h_s(ss_name)
            data["X"].extend([float(x) for x in _x_vals])
            data["S"].extend([float(s) for s in ss_s_vals])
            data["H"].extend([float(h) for h in ss_h_vals])
            data["Phase Name"].extend([ss_name] * len(_x_vals))

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            import pandas as pd

            return pd.DataFrame(data)
        raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")


# def _linear_ref_expr(enthalpy_jmol: float, transition_temp_k: float) -> sp.Expr:
#     if enthalpy_jmol == 0 or transition_temp_k == 0:
#         return sp.Integer(0)
#     return float(enthalpy_jmol) - t_sym * float(enthalpy_jmol) / float(transition_temp_k)


# def query_element_refs_from_mp(
#     components: list[str],
#     api_key: str,
# ) -> dict[str, dict[str, float | str | int]]:
#     if MPRester is None:
#         raise RuntimeError("mp_api is not installed; cannot query Materials Project.")
#     if not api_key:
#         raise RuntimeError("Missing MP API key for Materials Project query.")

#     refs: dict[str, dict[str, float | str | int]] = {}
#     with MPRester(api_key) as mpr:
#         for el in components:
#             docs = list(
#                 mpr.materials.summary.search(
#                     elements=[el],
#                     num_elements=(1, 1),
#                     fields=["material_id", "formula_pretty", "energy_per_atom", "energy_above_hull", "symmetry"],
#                     deprecated=False,
#                 )
#             )
#             docs = [
#                 doc
#                 for doc in docs
#                 if doc.energy_per_atom is not None
#                 and doc.symmetry is not None
#                 and doc.symmetry.number is not None
#             ]
#             if not docs:
#                 raise RuntimeError(f"No MP summary docs found for pure element '{el}'.")

#             ground = min(
#                 docs,
#                 key=lambda doc: (
#                     float(doc.energy_above_hull if doc.energy_above_hull is not None else 1e9),
#                     float(doc.energy_per_atom),
#                 ),
#             )
#             bcc_docs = [doc for doc in docs if int(doc.symmetry.number) == 229]
#             if not bcc_docs:
#                 raise RuntimeError(f"No BCC (spacegroup 229) MP entry found for pure element '{el}'.")
#             bcc = min(bcc_docs, key=lambda doc: float(doc.energy_per_atom))

#             delta_e_ev = float(bcc.energy_per_atom) - float(ground.energy_per_atom)
#             refs[el] = {
#                 "source": "mp-query",
#                 "ground_material_id": str(ground.material_id),
#                 "ground_spacegroup": int(ground.symmetry.number),
#                 "ground_symbol": str(ground.symmetry.symbol),
#                 "ground_energy_ev_per_atom": float(ground.energy_per_atom),
#                 "bcc_material_id": str(bcc.material_id),
#                 "bcc_spacegroup": int(bcc.symmetry.number),
#                 "bcc_symbol": str(bcc.symmetry.symbol),
#                 "bcc_energy_ev_per_atom": float(bcc.energy_per_atom),
#                 "delta_e_ev": float(delta_e_ev),
#                 "delta_e_jmol": float(delta_e_ev * EV_PER_ATOM_TO_J_PER_MOL),
#             }
#     return refs




def load_local_dft_convex_hull(components: list[str], entries_path: Path):
    entry_dicts = json.loads(entries_path.read_text(encoding="utf-8"))
    entries = [ComputedEntry.from_dict(entry) for entry in entry_dicts]
    if any(len(Composition(comp).elements) > 1 for comp in components):
        return CompoundPhaseDiagram(
            terminal_compositions=[Composition(comp) for comp in components],
            entries=entries,
        )
    return PhaseDiagram(elements=[Element(comp) for comp in components], entries=entries)



def auto_chg_temps_k(system: SolidSolutionBinaryLiquid) -> list[float]:
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    solid_phases = [
        phase
        for phase in system.phases
        if phase["name"] not in system.components + ["L"] and len(phase["points"]) > 0
    ]

    if solid_phases:
        t_high = max(max(phase["points"], key=lambda point: point[1])[1] for phase in solid_phases)
    elif system.phases[-1]["points"]:
        t_high = max(system.phases[-1]["points"], key=lambda point: point[1])[1]
    else:
        t_high = max(component[1] for component in system.component_data.values() if component[1] > 0)
    return [0.0, float(t_high)]


def build_chg_with_solid_solution(
    system: SolidSolutionBinaryLiquid,
    t_vals_k: list[float],
    hide_unstable_solution_curves: bool = False,
) -> go.Figure:
    pdp = PDPlotter(system.dft_ch)
    hull_fig = pdp.get_plot()
    params = system.get_params()

    stable_solution_labels: set[str] | None = None
    if hide_unstable_solution_curves:
        if not system.phases[-1]["points"]:
            system.update_phase_points()
        _, final_phases, _, _ = system.hsx.compute_tx()
        stable_labels = {str(label) for phase_list in final_phases for label in phase_list}
        stable_solution_labels = {"L", *system.ss_names}.intersection(stable_labels)

    traces: list[go.Scatter] = []
    for temp_k in reversed(t_vals_k):
        g_liq_expr = system.eqs["g_liquid"].subs(
            {
                t_sym: temp_k,
                a_sym: params[0],
                b_sym: params[1],
                c_sym: params[2],
                d_sym: params[3],
            }
        )
        g_liq_mid = sp.lambdify(xb_sym, g_liq_expr, "numpy")(_x_vals[1:-1]) if g_liq_expr.has(xb_sym) else []
        g_liq_mid = np.asarray(g_liq_mid, dtype=float)
        ga = float(system.eqs["ga"].subs({t_sym: temp_k}))
        gb = float(system.eqs["gb"].subs({t_sym: temp_k}))
        y_liq = np.array([ga, *g_liq_mid.tolist(), gb], dtype=float) / EV_PER_ATOM_TO_J_PER_MOL
        if not hide_unstable_solution_curves or (stable_solution_labels is not None and "L" in stable_solution_labels):
            traces.append(
                go.Scatter(
                    x=_x_vals,
                    y=y_liq,
                    mode="lines",
                    line={"width": 2.5},
                    name=f"Liquid {int(round(temp_k))} K",
                )
            )

        x_ss = np.asarray(_x_vals, dtype=float)
        for ss_name in system.ss_names:
            if hide_unstable_solution_curves and (
                stable_solution_labels is not None and ss_name not in stable_solution_labels
            ):
                continue
            y_ss = system.solid_solution_gibbs(ss_name, x_ss, temp_k) / EV_PER_ATOM_TO_J_PER_MOL
            traces.append(
                go.Scatter(
                    x=x_ss,
                    y=y_ss,
                    mode="lines",
                    line={"width": 2.5, "dash": "dash"},
                    name=f"{ss_name} {int(round(temp_k))} K",
                )
            )

    fig = go.Figure(data=traces + list(hull_fig.data), layout=hull_fig.layout)
    fig.update_layout(
        title=f"{system.sys_name} Convex Hull + Solution G Curves",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=620,
    )
    fig.update_xaxes(title="Composition (fraction)")
    return fig


def _split_segments(x_vals: np.ndarray, y_vals: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if x_vals.size <= 1:
        return [(x_vals, y_vals)]
    dx = np.diff(x_vals)
    positive_dx = dx[dx > 0]
    step = float(np.median(positive_dx)) if positive_dx.size else 1.0
    gap_threshold = max(1.5 * step, 0.8)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for idx in range(1, x_vals.size):
        if x_vals[idx] - x_vals[idx - 1] > gap_threshold:
            segments.append((x_vals[start:idx], y_vals[start:idx]))
            start = idx
    segments.append((x_vals[start:], y_vals[start:]))
    return segments


def _phase_display_name(system: SolidSolutionBinaryLiquid, phase_label: str) -> str:
    if phase_label in system.ss_names:
        return f"({system.components[0]}, {system.components[1]})"
    return phase_label


def _phase_color(system: SolidSolutionBinaryLiquid, phase_label: str) -> str:
    if phase_label == "L":
        return "cornflowerblue"
    if phase_label in system.ss_names:
        ss_palette = ["#2f9e44", "#0b7285", "#e67700", "#5f3dc4"]
        idx = system.ss_names.index(phase_label) % len(ss_palette)
        return ss_palette[idx]
    if phase_label in system.components:
        return "#222222"
    return "#555555"


def build_tx_with_solid_solution(
    system: SolidSolutionBinaryLiquid,
    show_tie_lines: bool = True,
    tie_line_stride: int = 1,
) -> go.Figure:
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    df_tx, final_phases, simplices, temps_k = system.hsx.compute_tx()
    temps_c = temps_k - 273.15
    df = df_tx.copy()
    df["x_at"] = df["x"].astype(float) * 100.0
    df["t_c"] = df["t"].astype(float) - 273.15

    all_t_candidates = [float(np.min(temps_c)), float(np.max(temps_c))]
    if system.digitized_liq:
        liq_t = [float(point[1] - 273.15) for point in system.digitized_liq]
        all_t_candidates.extend([min(liq_t), max(liq_t)])
    y_lo = min(float(system.temp_range[0] - 273.15), min(all_t_candidates))
    y_hi = max(float(system.temp_range[1] - 273.15), max(all_t_candidates))
    y_pad = max(40.0, 0.04 * (y_hi - y_lo))
    y_lo -= y_pad
    y_hi += y_pad

    fig = go.Figure()

    if show_tie_lines and len(simplices) > 0:
        seen: set[tuple[float, float, float]] = set()
        stride = max(1, int(tie_line_stride))
        for idx, simplex in enumerate(simplices):
            if idx % stride != 0:
                continue
            x_vals = system.hsx.points[simplex, 0].astype(float) * 100.0
            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            if x_max - x_min < 0.25:
                continue
            t_c = float(temps_c[idx])
            key = (round(t_c, 3), round(x_min, 3), round(x_max, 3))
            if key in seen:
                continue
            seen.add(key)
            includes_liquid = any(label == "L" for label in final_phases[idx])
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[t_c, t_c],
                    mode="lines",
                    line={
                        "color": "rgba(120,120,120,0.30)" if includes_liquid else "rgba(170,170,170,0.22)",
                        "width": 1.0 if includes_liquid else 0.8,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    labels = [str(label) for label in sorted(df["label"].unique().tolist())]
    if "L" in labels:
        labels = [label for label in labels if label != "L"] + ["L"]

    for label in labels:
        phase_df = df[df["label"] == label]
        if phase_df.empty:
            continue

        x_vals = phase_df["x_at"].to_numpy(dtype=float)
        t_vals = phase_df["t_c"].to_numpy(dtype=float)

        # Build composition-wise envelopes (min/max T) and keep disconnected ranges separate.
        envelope: dict[float, list[float]] = {}
        for x_val, t_val in zip(x_vals, t_vals):
            x_key = round(float(x_val), 6)
            if x_key not in envelope:
                envelope[x_key] = [float(t_val), float(t_val)]
            else:
                envelope[x_key][0] = min(envelope[x_key][0], float(t_val))
                envelope[x_key][1] = max(envelope[x_key][1], float(t_val))

        xs = np.array(sorted(envelope.keys()), dtype=float)
        t_min = np.array([envelope[x][0] for x in xs], dtype=float)
        t_max = np.array([envelope[x][1] for x in xs], dtype=float)

        color = _phase_color(system, label)
        name = _phase_display_name(system, label)

        if xs.size == 1:
            x0 = float(xs[0])
            y0 = float(np.min(t_vals))
            y1 = float(np.max(t_vals))
            if label in system.components and (x0 < 0.5 or x0 > 99.5):
                y0 = y_lo
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0],
                    y=[y0, y1],
                    mode="lines",
                    line={"color": color, "width": 2.4},
                    name=name,
                    showlegend=(label != "L"),
                )
            )
            continue

        lower_segments = _split_segments(xs, t_min)
        show_legend = label != "L"
        for seg_x, seg_y in lower_segments:
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    line={"color": color, "width": 2.8 if label == "L" else 2.4},
                    name=name,
                    showlegend=show_legend,
                )
            )
            show_legend = False

    if system.digitized_liq:
        fig.add_trace(
            go.Scatter(
                x=[float(point[0] * 100.0) for point in system.digitized_liq],
                y=[float(point[1] - 273.15) for point in system.digitized_liq],
                mode="lines",
                line={"color": "#b82e2e", "width": 2.0, "dash": "dash"},
                name="Assessed Liquidus",
            )
        )

    for ss_name in system.ss_names:
        ss_phase = next((phase for phase in system.phases if phase["name"] == ss_name), None)
        ss_points = np.array(ss_phase["points"], dtype=float) if ss_phase is not None else np.empty((0, 2))
        if ss_points.size > 0:
            ss_x = ss_points[:, 0] * 100.0
            ss_t = ss_points[:, 1] - 273.15
            mid_idx = int(np.argmax(ss_t))
            fig.add_annotation(
                x=float(ss_x[mid_idx]),
                y=float(ss_t[mid_idx] - 0.08 * (y_hi - y_lo)),
                text=ss_name,
                showarrow=False,
                font={"size": 14, "color": "black"},
                textangle=-90,
            )

        ss_color = _phase_color(system, ss_name)
        ss_ref_endpoints = [
            (0.0, float(system.component_data[system.components[0]]["T_fusion"]) - 273.15),
            (100.0, float(system.component_data[system.components[1]]["T_fusion"]) - 273.15),
        ]
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in ss_ref_endpoints],
                y=[point[1] for point in ss_ref_endpoints],
                mode="markers",
                marker={"color": ss_color, "size": 8},
                name=f"{ss_name} endpoint refs",
                showlegend=False,
            )
        )

    fig.add_annotation(
        x=50,
        y=y_hi - 0.08 * (y_hi - y_lo),
        text="L",
        showarrow=False,
        font={"size": 15, "color": "black"},
    )
    fig.add_annotation(
        x=-0.05,
        y=-0.09,
        xref="paper",
        yref="paper",
        text=system.components[0],
        showarrow=False,
        font={"color": "black", "size": 14},
        xanchor="left",
        yanchor="middle",
    )
    fig.add_annotation(
        x=1.05,
        y=-0.09,
        xref="paper",
        yref="paper",
        text=system.components[1],
        showarrow=False,
        font={"color": "black", "size": 14},
        xanchor="right",
        yanchor="middle",
    )

    fig.update_layout(
        title=f"<b>{system.sys_name} Solid-Solution HSX Phase Diagram</b>",
        xaxis={"range": [0, 100], "title": "Composition (at. %)"},
        yaxis={"range": [y_lo, y_hi], "title": "Temperature (°C)"},
        width=900,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        margin={"t": 72, "b": 72, "r": 40},
    )
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor="black",
        linewidth=1.5,
        ticks="outside",
        tickcolor="black",
        ticklen=5,
        tickwidth=1,
        minor={"tickcolor": "black", "ticklen": 2, "tickwidth": 1, "nticks": 5},
        minor_ticks="outside",
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor="black",
        linewidth=1.5,
        ticks="outside",
        tickcolor="black",
        ticklen=5,
        tickwidth=1,
        minor={"tickcolor": "black", "ticklen": 2, "tickwidth": 1, "nticks": 5},
        minor_ticks="outside",
    )
    return fig


def build_fit_summary(system: SolidSolutionBinaryLiquid, best_fit: dict) -> dict:
    return {
        "system": system.sys_name,
        "mae": float(best_fit["mae"]),
        "rmse": float(best_fit["rmse"]),
        "mape": float(best_fit["mape"]),
        "params": {
            "L0_a": float(best_fit["L0_a"]),
            "L0_b": float(best_fit["L0_b"]),
            "L1_a": float(best_fit["L1_a"]),
            "L1_b": float(best_fit["L1_b"]),
        },
        "ss_models": {
            ss_name: {
                "ref_mode": str(ss_model["ref_mode"]),
                "omega_ev": float(ss_model["omega_ev"]),
                "omega_jmol": float(ss_model["omega_jmol"]),
                "offset_a_ev": float(ss_model["offset_a_ev"]),
                "offset_b_ev": float(ss_model["offset_b_ev"]),
                "offset_a_jmol": float(ss_model["offset_a_jmol"]),
                "offset_b_jmol": float(ss_model["offset_b_jmol"]),
                "refs": ss_model["refs"],
            }
            for ss_name, ss_model in system.ss_models.items()
        },
    }


def main():

    # elts = ["Zr", "Hf", "Nb", "W", "Al"]
    # system_combos = [sorted(elts[i], elts[j]) for i in range(len(elts)) for j in range(i + 1, len(elts))]

    sys_name = "Nb-W"
    system = SolidSolutionBinaryLiquid.from_cache(sys_name, pd_ind=0, ref_mode='omegas-legacy', param_format='comb-exp',
                                                  omegas_path=cfg.data_dir / "omegas_hcp.json")

    print("Solid-solution models:")
    for ss_name, ss_model in system.ss_models.items():
        print(
            f"  phase={ss_name}",
            f"omega={ss_model['omega_ev']:.6f} eV/atom",
            f"offsets=({ss_model['offset_a_ev']:.6f}, {ss_model['offset_b_ev']:.6f}) eV/atom",
            f"ref_mode={ss_model['ref_mode']}",
        )
        print("  Pure-element references:")
        for el in system.components:
            ref = ss_model["refs"][el]
            print(
                f"    {el}: ground={ref['ground_material_id']} SG={ref['ground_spacegroup']} {ref['ground_symbol']} "
                f"E={ref['ground_energy_ev_per_atom']:.6f} eV/atom | "
                f"SS-ref={ref['material_id']} SG={ref['spacegroup']} {ref['symbol']} "
                f"E={ref['energy_ev_per_atom']:.6f} eV/atom | "
                f"delta={ref['delta_e_ev']:.6f} eV/atom"
            )
        print()

    fit_results = system.fit_parameters(
        verbose=True,
        n_opts=10,
        max_iter=128,
        disable_inv_constrs=True,
        check_full_ss=False,
    )

    if not fit_results:
        raise RuntimeError("No fit results were produced. Check input data and fitting constraints.")

    best_fit = min(fit_results, key=lambda entry: entry.get("mae", float("inf")))
    print("\n--- Best Fit ---")
    print(f"MAE   : {best_fit['mae']:.3f} K")
    print(f"RMSE  : {best_fit['rmse']:.3f} K")
    print(f"MAPE  : {best_fit['mape']:.3f} %")
    print(f"Params: L0_a={best_fit['L0_a']:.3f}, L0_b={best_fit['L0_b']:.6f}, "
          f"L1_a={best_fit['L1_a']:.3f}, L1_b={best_fit['L1_b']:.6f}")

    summary = build_fit_summary(system, best_fit)
    summary_path = cfg.project_root / "dev" / "data" / f"{sys_name}_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    fit_fig = build_tx_with_solid_solution(
        system
    )
    # fit_path = cfg.project_root / "figures" / f"{sys_name}_fit_plus_liq.html"
    # fit_fig.write_html(str(fit_path), include_plotlyjs="cdn")
    # print(f"Saved plot: {fit_path}")
    fit_fig.show()

    t_vals_k = auto_chg_temps_k(system)
    chg_fig = build_chg_with_solid_solution(system, t_vals_k=t_vals_k)
    chg_fig.show()
    chg_fig_hidden = build_chg_with_solid_solution(system, t_vals_k=t_vals_k, hide_unstable_solution_curves=True)
    chg_fig_hidden.show()
    # chg_path = cfg.project_root / "figures" / f"{sys_name}_chg_with_ss.html"
    # chg_fig.write_html(str(chg_path), include_plotlyjs="cdn")
    # print(f"Saved plot: {chg_path}")

    # plotter = BLPlotter(system)
    # nmp_path = cfg.project_root / "figures" / f"{sys_name}_nmp.png"
    # plotter.write_image("nmp", str(nmp_path), image_format="png")
    # print(f"Saved plot: {nmp_path}")
    # plotter.show("nmp")


if __name__ == "__main__":
    main()
