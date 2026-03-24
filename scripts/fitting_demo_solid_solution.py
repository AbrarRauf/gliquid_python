from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp

import gliquid.config as cfg

# WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("C:\\Users\\AbrarRauf\\University of Michigan Dropbox\\Abrar Rauf\\WHSun_Lab\\G_liquid")
print(f"Setting workspace root to: {WORKSPACE_ROOT}")

cfg.set_project_root(WORKSPACE_ROOT)
cfg.set_data_dir(Path(cfg.project_root / "matrix_data"))
cfg.set_dir_structure(cfg._DIR_STRUCT_OPTS[1])
from ternary_interpolation.auth import mpapi_key
os.environ["MP_API_KEY"] = mpapi_key

import gliquid.load_binary_data as lbd
from gliquid.hsx import HSX
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
    build_thermodynamic_expressions,
)  
from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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
DEFAULT_REF_MODE = "omegas-legacy"  # Options: "binary-cache", "omegas-legacy", or "element-db"
DEFAULT_OMEGAS_PATH = cfg.data_dir / "omegas.json"

# Fixed colors for solution phases; these are reserved and never reused for hull phases.
SS_FIXED_COLORS = {
    "BCC": "#d7263d",
    "FCC": "#1b9aaa",
    "HCP": "#f4a259",
}

def _safe_spacegroup(structure) -> tuple[int | None, str | None]:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        return int(sga.get_space_group_number()), str(sga.get_space_group_symbol())
    except Exception as e:
        print(e)
        return None, None


def get_dft_ch_entries(input, dft_type='GGA',
                       verbose=False) -> list[ComputedStructureEntry]:
    """
    Returns the DFT convex hull of a given system with specified functionals.

    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        dft_type (str): Functional type, e.g., 'GGA', 'GGA/GGA+U', etc.
        inc_structure_data (bool): Whether to include structural data.
        verbose (bool): Whether to print detailed output.

    Returns:
        A tuple of the phase diagram and a dictionary of stable entry atomic volumes.
    """
    components, sys_name, _ = lbd.validate_and_format_binary_system(input)

    supp_dft_types = ["GGA", "R2SCAN", "MIXED"]
    if dft_type not in supp_dft_types:
        raise SyntaxError(
            f"dft_type '{dft_type}' is not currently supported! "
            f"Please specify as one of the following: {', '.join(supp_dft_types)}"
        )
    if verbose:
        print(f"Using DFT entries solved with {dft_type} functionals.")

    if cfg.dir_structure == 'nested':
        sys_dir = os.path.join(cfg.data_dir, sys_name)
        os.makedirs(sys_dir, exist_ok=True)
    elif cfg.dir_structure == 'flat':
        sys_dir = cfg.data_dir
    else:
        raise ValueError(f"Invalid dir_structure '{cfg.dir_structure}'. Must be 'nested' or 'flat'.")
    
    dft_entries_file = os.path.join(sys_dir, f"{sys_name}_ENTRIES_MP_{dft_type}.json")

    # Yb-containing structures are only available with R2SCAN functional
    # See https://docs.materialsproject.org/changes/database-versions#v2023.11.1
    # and https://docs.materialsproject.org/changes/database-versions#v2025.02.12
    if 'Yb' in components and not os.path.exists(dft_entries_file):
        print("Warning: Yb-containing structures are only available with R2SCAN or MIXED functionals on the MP database.") 
        # dft_type = 'R2SCAN' # optional, uncomment these lines to enforce R2SCAN functionals for Yb systems
        # dft_entries_file = os.path.join(sys_dir, f"{sys_name}_ENTRIES_MP_{dft_type}.json")

    query_mp = True
    if os.path.exists(dft_entries_file):
        query_mp = False
        with open(dft_entries_file, "r") as f:
            computed_entry_dicts = json.load(f)
        if verbose:
            print("Loading cached DFT entry data.")
    if not any(c.get('data', None) is None for c in computed_entry_dicts):
        query_mp = True

    if query_mp:
        computed_entry_dicts = lbd._get_dft_entries_from_components(components, dft_type, keep_data=True)
        if verbose:
            print(f"Caching DFT entry data as {dft_entries_file}...")
        with open(dft_entries_file, "w") as f:
            json.dump(computed_entry_dicts, f)
    return [ComputedStructureEntry.from_dict(entry_dict) for entry_dict in computed_entry_dicts]


def _find_gs_polymorph(polymorphs: list[dict[str, any]]) -> dict[str, any] | None:
    for p in polymorphs:
        if p["delta_H_J_per_mol"] == 0:
            return p
    return None

def _find_matching_polymorph(phase, polymorphs: dict[str, list[any]]) -> dict[str, any] | None:
    for p in polymorphs:
        if p["spacegroup_number"] == SS_SGS[phase]:
            if p["spacegroup_symbol"] != SS_SYMBOLS[phase]:
                print("Warning: spacegroup symbol does not match expected for phase '{phase}'. Check for inconsistencies in the omegas file.")
            return p
    return None

def _compute_solid_ss_entropy(
    el: str,
    ss_phase: str,
    delta_h_jmol: float,
    phase_refs_so_far: dict[str, dict[str, dict]],
    phase_transitions: dict[str, dict],
) -> float:
    """
    Compute the cumulative entropy for a solid SS phase using the stepwise
    formula S = Σ ΔH_i / T_i over all SS phases with known transition
    temperatures at or below this phase's transition temperature.

    phase_refs_so_far may be partially populated — only phases already
    resolved and present in it are included in the sum.
    """
    elem_pt = phase_transitions.get(el, {})
    phase_pt = next(
        (p for p in elem_pt.get("phases", [])
         if p.get("phase_type") == "solid"
         and p.get("spacegroup_number") == SS_SGS.get(ss_phase)),
        None,
    )
    t_this = phase_pt.get("transition_temperature_K") if phase_pt else None
    if t_this is None or t_this <= 0:
        return 0.0

    # Gather all SS steps with known T <= t_this, including this phase itself
    candidates: list[tuple[float, float]] = []
    for phase, refs in phase_refs_so_far.items():
        if el not in refs:
            continue
        p_pt = next(
            (p for p in elem_pt.get("phases", [])
             if p.get("phase_type") == "solid"
             and p.get("spacegroup_number") == SS_SGS.get(phase)),
            None,
        )
        t_p = p_pt.get("transition_temperature_K") if p_pt else None
        if t_p is not None and 0 < t_p <= t_this:
            candidates.append((refs[el]["delta_h_jmol"], t_p))

    # Include this phase itself
    candidates.append((delta_h_jmol, t_this))
    candidates.sort(key=lambda x: x[1])

    s_accum = 0.0
    prev_h = 0.0
    for dh, t in candidates:
        s_accum += (dh - prev_h) / t
        prev_h = dh

    return s_accum

def _make_ground_ref(
    source: str,
    material_id: str,
    spacegroup: int,
    symbol: str,
    energy_ev_per_atom: float,
) -> dict[str, float | str | int]:
    return {
        "source": source,
        "ground_material_id": material_id,
        "ground_spacegroup": spacegroup,
        "ground_symbol": symbol,
        "ground_energy_ev_per_atom": energy_ev_per_atom,
    }


def _make_phase_ref(
    ss_phase: str,
    material_id: str,
    energy_ev_per_atom: float,
    delta_h_jmol: float,
    delta_s_jmol_k: float,
    spacegroup: int | None = None,
    symbol: str | None = None,
) -> dict[str, float | str | int]:
    return {
        "material_id": material_id,
        "spacegroup": spacegroup if spacegroup is not None else SS_SGS.get(ss_phase, -1),
        "symbol": symbol if symbol is not None else SS_SYMBOLS.get(ss_phase, "unknown"),
        "energy_ev_per_atom": energy_ev_per_atom,
        "delta_h_jmol": delta_h_jmol,
        "delta_s_jmol_k": delta_s_jmol_k,
    }


def _build_ss_models_from_refs(
    components: list[str],
    ground_refs: dict[str, dict],
    phase_refs: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Merge ground_refs and phase_refs into the ss_models skeleton."""
    ss_models: dict[str, dict] = {}
    for ss_phase, el_refs in phase_refs.items():
        refs = {el: dict(ground_refs[el]) for el in components}
        for el, pr in el_refs.items():
            refs[el].update(pr)
        ss_models[ss_phase] = {"refs": refs}
    return ss_models


def _package_ss_models(
    ss_models: dict[str, dict],
    components: list[str],
    omega_data: dict[str, dict[str, float]],
    pair_key: str,
    ref_mode: str,
) -> dict[str, dict]:
    """
    Attach omega and delta H/S fields to every phase in ss_models in-place.
    Raises KeyError if omega is missing for a phase that has refs.
    """
    for ss_phase, model_dict in ss_models.items():
        if "refs" not in model_dict:
            continue
        omega_block = omega_data.get(ss_phase, {})
        if pair_key not in omega_block:
            raise KeyError(
                f"Could not find omega for pair '{pair_key}' in phase '{ss_phase}'."
            )
        refs = model_dict["refs"]
        model_dict.update({
            "ref_mode": ref_mode,
            "omega_jmol": float(omega_block[pair_key]) * EV_PER_ATOM_TO_J_PER_MOL,
            "deltaH_a_jmol": float(refs[components[0]]["delta_h_jmol"]),
            "deltaH_b_jmol": float(refs[components[1]]["delta_h_jmol"]),
            "deltaS_a_jmol_k": float(refs[components[0]]["delta_s_jmol_k"]),
            "deltaS_b_jmol_k": float(refs[components[1]]["delta_s_jmol_k"]),
        })
    return ss_models


def _resolve_refs_legacy(
    data: dict,
    components: list[str],
    component_data: dict | None,
    phase_transitions: dict[str, dict] | None,
) -> tuple[dict, dict]:
    """
    Build ground_refs and phase_refs from the flat omegas file (legacy format).

    Entropy reference: recalculated from transition temperature when
    component_data is available and not in legacy list format.
    """
    element_blocks: dict[str, dict[str, float]] = data["elements"]

    # Ground state = minimum energy across all phases listed in the file
    stable_pure: dict[str, float] = {}
    for el in components:
        candidates = [float(block[el]) for block in element_blocks.values() if el in block]
        if not candidates:
            raise KeyError(
                f"Could not find pure-element references for '{el}' in omegas file."
            )
        stable_pure[el] = min(candidates)

    ground_refs: dict[str, dict] = {
        el: _make_ground_ref(
            source="omegas-legacy",
            material_id="legacy",
            spacegroup=-1,
            symbol="legacy",
            energy_ev_per_atom=stable_pure[el],
        )
        for el in components
    }

    phase_refs: dict[str, dict[str, dict]] = {}
    for ss_phase, phase_block in element_blocks.items():
        if not all(el in phase_block for el in components):
            continue
        phase_refs[ss_phase] = {}
        for el in components:
            ss_e = float(phase_block[el])
            delta_h_jmol = (ss_e - stable_pure[el]) * EV_PER_ATOM_TO_J_PER_MOL

            delta_s_jmol_k = 0.0
            if component_data:
                poly = _find_matching_polymorph(ss_phase, component_data[el].get("polymorphs", {}))
                t_trans = poly.get("transition_temperature_K", 0) if poly else 0
                if t_trans > 0:
                    delta_s_jmol_k = _compute_solid_ss_entropy(el, ss_phase, delta_h_jmol, phase_refs, phase_transitions)

            phase_refs[ss_phase][el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id="legacy",
                energy_ev_per_atom=ss_e,
                delta_h_jmol=delta_h_jmol,
                delta_s_jmol_k=delta_s_jmol_k,
            )

    return ground_refs, phase_refs


def _resolve_refs_cache(
    components: list[str],
    entries: list[ComputedStructureEntry],
) -> tuple[dict, dict]:
    """
    Build ground_refs and phase_refs from pymatgen ComputedStructureEntries.

    Entropy reference: set to zero (DFT gives no finite-T entropy correction).
    """
    ground_refs: dict[str, dict] = {}
    phase_refs: dict[str, dict[str, dict]] = {}

    for el in components:
        pure_entries = [e for e in entries if e.composition.reduced_formula == el]
        if not pure_entries:
            raise RuntimeError(f"No pure-element entries found for '{el}'.")

        ground = min(pure_entries, key=lambda e: float(e.energy_per_atom))
        ground_sg, ground_symbol = _safe_spacegroup(getattr(ground, "structure", None))

        ground_refs[el] = _make_ground_ref(
            source="binary-cache",
            material_id=str(getattr(ground, "entry_id", "unknown")),
            spacegroup=int(ground_sg) if ground_sg is not None else -1,
            symbol=ground_symbol or "unknown",
            energy_ev_per_atom=float(ground.energy_per_atom),
        )

        for ss_phase in ALL_SS_PHASES:
            phase_entries = [
                e for e in pure_entries
                if _safe_spacegroup(getattr(e, "structure", None))[0] == SS_SGS[ss_phase]
            ]
            if not phase_entries:
                print(
                    f"No {ss_phase} (spacegroup {SS_SGS[ss_phase]}) pure entry "
                    f"found for '{el}' in local cache."
                )
                continue

            best = min(phase_entries, key=lambda e: float(e.energy_per_atom))
            sg, symbol = _safe_spacegroup(getattr(best, "structure", None))
            delta_h_jmol = (float(best.energy_per_atom) - float(ground.energy_per_atom)) \
                           * EV_PER_ATOM_TO_J_PER_MOL

            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id=str(getattr(best, "entry_id", "unknown")),
                energy_ev_per_atom=float(best.energy_per_atom),
                delta_h_jmol=delta_h_jmol,
                delta_s_jmol_k=0.0,
                spacegroup=int(sg) if sg is not None else SS_SGS[ss_phase],
                symbol=symbol or SS_SYMBOLS[ss_phase],
            )

    return ground_refs, phase_refs


def _resolve_refs_db(
    components: list[str],
    component_data: dict[str, dict],
) -> tuple[dict, dict]:
    """
    Build ground_refs and phase_refs from the element-db (component_data).

    Entropy reference: taken directly from delta_S_J_per_mol_K in the polymorph
    entry when present; falls back to 0.0 for legacy component_data that lacks it.
    """
    ground_refs: dict[str, dict] = {}
    phase_refs: dict[str, dict[str, dict]] = {}

    for el in components:
        polymorphs = component_data[el].get("polymorphs", {})
        p_ground = _find_gs_polymorph(polymorphs)

        ground_refs[el] = _make_ground_ref(
            source="element-db",
            material_id=p_ground.get("materials_project_id", "unknown") if p_ground else "unknown",
            spacegroup=p_ground.get("spacegroup_number", -1) if p_ground else -1,
            symbol=p_ground.get("spacegroup_symbol", "unknown") if p_ground else "unknown",
            energy_ev_per_atom=(
                p_ground.get("enthalpy_J_per_mol", -EV_PER_ATOM_TO_J_PER_MOL)
                / EV_PER_ATOM_TO_J_PER_MOL
            ) if p_ground else -1.0,
        )

        for ss_phase in ALL_SS_PHASES:
            p_poly = _find_matching_polymorph(ss_phase, polymorphs)
            if p_poly is None:
                continue
            phase_refs.setdefault(ss_phase, {})[el] = _make_phase_ref(
                ss_phase=ss_phase,
                material_id=p_poly.get("materials_project_id", "unknown"),
                energy_ev_per_atom=p_poly.get(
                    "enthalpy_J_per_mol", -EV_PER_ATOM_TO_J_PER_MOL
                ) / EV_PER_ATOM_TO_J_PER_MOL,
                delta_h_jmol=p_poly.get("delta_H_J_per_mol", -1.0),
                # Falls back to 0.0 when key is absent (legacy component_data)
                delta_s_jmol_k=p_poly.get("delta_S_J_per_mol_K", 0.0),
                spacegroup=p_poly.get("spacegroup_number", SS_SGS[ss_phase]),
                symbol=p_poly.get("spacegroup_symbol", SS_SYMBOLS[ss_phase]),
            )

    return ground_refs, phase_refs

def _accumulate_entropy(
    el: str,
    ordered_solid_steps: list[tuple[float, float]],
    phase_transitions: dict[str, dict],
) -> tuple[float, float]:
    """
    Compute cumulative (H_liq, S_liq) by stepwise accumulation of ΔS = ΔH / T
    over ordered solid steps up to and including the fusion step.

    Args:
        el: Element symbol.
        ordered_solid_steps: List of (delta_h_jmol, transition_temperature_K)
            for each solid phase below the melt, ordered by increasing
            transition temperature. delta_h_jmol is the resolver-produced
            cumulative enthalpy for that phase (relative to ground state).
        phase_transitions: The elements dict from phase_transitions.json.

    Returns:
        (H_liq, S_liq) where S_liq = Σ ΔH_i/T_i over all steps including fusion.
    """
    elem_pt = phase_transitions.get(el, {})
    liquid_phase = next(
        (p for p in elem_pt.get("phases", []) if p["phase_type"] == "liquid"),
        None,
    )
    if liquid_phase is None:
        print(f"Warning: No liquid phase found for '{el}' in phase_transitions. H_liq/S_liq set to 0.")
        return 0.0, 0.0

    h_fusion = liquid_phase.get("delta_H_J_per_mol")
    t_melt = liquid_phase.get("transition_temperature_K")
    if h_fusion is None or t_melt is None or t_melt <= 0:
        print(f"Warning: Missing fusion enthalpy or melt temperature for '{el}'. H_liq/S_liq set to 0.")
        return 0.0, 0.0

    # Stepwise entropy accumulation over solid steps.
    # Each step's ΔH is the increment from the previous phase, not the
    # cumulative value, so we difference consecutive cumulative enthalpies.
    s_accum = 0.0
    prev_h = 0.0  # ground state: H = 0
    for delta_h_jmol, t_trans in ordered_solid_steps:
        delta_h_step = delta_h_jmol - prev_h
        s_accum += delta_h_step / t_trans
        prev_h = delta_h_jmol

    # Final fusion step
    h_liq = prev_h + h_fusion
    s_liq = s_accum + h_fusion / t_melt

    return h_liq, s_liq


def _get_ordered_solid_steps_from_phase_refs(
    el: str,
    phase_refs: dict[str, dict[str, dict]],
    phase_transitions: dict[str, dict],
    t_melt: float,
) -> list[tuple[float, float]]:
    """
    Build an ordered list of (delta_h_jmol, transition_temperature_K) for
    all SS phases that have a known transition temperature strictly below
    T_melt, ordered by increasing transition temperature.

    Transition temperatures are always taken from phase_transitions regardless
    of resolver, since omegas/cache sources don't provide temperatures.
    Only phases present in phase_refs are considered — intermediate phases
    in phase_transitions that are not SS phases are excluded.
    """
    elem_pt = phase_transitions.get(el, {})
    candidates: list[tuple[float, float]] = []

    for ss_phase, refs in phase_refs.items():
        if el not in refs:
            continue
        phase_pt = next(
            (p for p in elem_pt.get("phases", [])
             if p.get("phase_type") == "solid"
             and p.get("spacegroup_number") == SS_SGS.get(ss_phase)),
            None,
        )
        t_trans = phase_pt.get("transition_temperature_K") if phase_pt else None
        if t_trans is None or t_trans <= 0 or t_trans >= t_melt:
            continue
        candidates.append((refs[el]["delta_h_jmol"], t_trans))

    return sorted(candidates, key=lambda x: x[1])


def _reconcile_liquid_refs_in_component_data(
    components: list[str],
    component_data: dict[str, dict],
    phase_refs: dict[str, dict[str, dict]],
    phase_transitions: dict[str, dict],
) -> None:
    """
    Overwrite H_liq and S_liq in component_data using the general stepwise
    entropy accumulation formula:

        S_liq = Σ ΔH_i / T_i   (solid steps) + H_fusion / T_melt

    where ΔH_i are incremental enthalpies between consecutive phases and T_i
    are their transition temperatures. Solid steps use resolver-produced
    delta_h_jmol values; temperatures and fusion data come from
    phase_transitions.json.

    Only SS phases present in phase_refs are included in the solid steps —
    intermediate phases from phase_transitions that are not SS phases are
    excluded. If no SS phases fall below T_melt the ground state (H=S=0)
    is used as the sole prior step.

    Mutates component_data in-place.
    """
    for el in components:
        elem_pt = phase_transitions.get(el, {})
        t_melt = next(
            (p.get("transition_temperature_K") for p in elem_pt.get("phases", [])
             if p["phase_type"] == "liquid"),
            None,
        )
        if t_melt is None:
            print(f"Warning: No melt temperature found for '{el}'. Skipping reconciliation.")
            continue

        ordered_steps = _get_ordered_solid_steps_from_phase_refs(
            el, phase_refs, phase_transitions, t_melt
        )
        h_liq, s_liq = _accumulate_entropy(el, ordered_steps, phase_transitions)
        component_data[el]["H_liq"] = h_liq
        component_data[el]["S_liq"] = s_liq

def load_solid_solution_models(
    omegas_path: Path,
    components: list[str],
    component_data: dict[str, any],
    entries: list[ComputedStructureEntry],
    phase_transitions: dict[str, dict],
    ref_mode: str = DEFAULT_REF_MODE,
) -> dict[str, dict[str, float]]:

    legacy_component_data = False
    if any(isinstance(v, list) for v in component_data.values()):
        print(
            "Using component data from legacy format (list of dicts). "
            "Allotrope transition temperatures not considered."
        )
        legacy_component_data = True
    elif not all(isinstance(v, dict) for v in component_data.values()):
        raise ValueError(
            "component_data values must be either a list of dicts (legacy) "
            "or dict of dicts (element-db format)."
        )

    data = json.loads(omegas_path.read_text(encoding="utf-8"))
    pair_key = "-".join(sorted(components))
    omega_data: dict[str, dict[str, float]] = data.get("omegas", {})

    # --- resolve source-specific refs ---
    if ref_mode == "omegas-legacy":
        # Filter to only phases that have both an omega and element entries
        # for this pair before resolving, so phase_refs stays consistent.
        available_phases = {
            phase for phase, block in omega_data.items()
            if pair_key in block and phase in data.get("elements", {})
            and all(el in data["elements"][phase] for el in components)
        }
        if not available_phases:
            raise KeyError(
                f"No solid-solution phases found for pair '{pair_key}' "
                f"with matching omegas/elements entries."
            )
        ground_refs, phase_refs = _resolve_refs_legacy(
            data=data,
            components=components,
            component_data=component_data if not legacy_component_data else None,
            phase_transitions=phase_transitions,
        )
        # Keep only the phases that passed the availability check above
        phase_refs = {k: v for k, v in phase_refs.items() if k in available_phases}

    elif ref_mode == "binary-cache":
        ground_refs, phase_refs = _resolve_refs_cache(components, entries)

    elif ref_mode == "element-db":
        if legacy_component_data:
            raise ValueError("ref_mode 'element-db' requires non-legacy component_data format.")
        ground_refs, phase_refs = _resolve_refs_db(components, component_data)

    else:
        raise ValueError("ref_mode must be one of: binary-cache, omegas-legacy, element-db.")
    
    # Reconcile H_liq/S_liq in component_data using the resolver's own
    # solid delta_h values. Skipped for legacy format since it carries no
    # per-polymorph enthalpy data to derive a meaningful solid baseline from.
    if not legacy_component_data:
        _reconcile_liquid_refs_in_component_data(
            components, component_data, phase_refs, phase_transitions
        )
    else:
        print(
            "Skipping H_liq/S_liq reconciliation: legacy component_data format "
            "does not carry per-polymorph enthalpy data."
        )

    # --- shared assembly and packaging ---
    ss_models = _build_ss_models_from_refs(components, ground_refs, phase_refs)
    return _package_ss_models(ss_models, components, omega_data, pair_key, ref_mode)



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
                {'name': ss_name, 'is_solution': True, 'points': []},
            )
            insert_idx += 1

    @classmethod
    def from_cache(cls, input, **kwargs) -> SolidSolutionBinaryLiquid:
        """
        Create a SolidSolutionBinaryLiquid instance from cached data.
        This class method loads binary liquid data from cache and constructs a 
        SolidSolutionBinaryLiquid object with solid solution models. It filters 
        phases to include only intermediate solid solutions (excluding pure endpoints 
        where composition = 0 or 1).
        Args:
            input: Input identifier or path for loading cached binary liquid data.
            **kwargs: Arbitrary keyword arguments including:
                ref_mode (str): Reference mode for calculations. Defaults to DEFAULT_REF_MODE.
                omegas_path (str): Path to omega parameters. Defaults to DEFAULT_OMEGAS_PATH.
                verbose (bool): Enable verbose output. Defaults to False.
                Additional arguments passed to parent class initialization.
        Returns:
            SolidSolutionBinaryLiquid: Initialized instance with loaded solid solution models
                and phases filtered to include only intermediate compositions.
        """

        bl = BinaryLiquid.from_cache(input, **kwargs)
        ref_mode = kwargs.get("ref_mode", DEFAULT_REF_MODE)

        ss_models = load_solid_solution_models(
            components=bl.components,
            component_data=bl.component_data,
            entries=get_dft_ch_entries(bl.components, dft_type=bl.dft_type, verbose=kwargs.get("verbose", False)),
            omegas_path=kwargs.get("omegas_path", DEFAULT_OMEGAS_PATH),
            phase_transitions=getattr(lbd, "phase_transitions", None),
            ref_mode=ref_mode,
        )

        print("Updated component data with reconciled liquid references:")
        for comp, data in bl.component_data.items():
            print(f"{comp}: H_liq = {data['H_liq']} J/mol, S_liq = {data['S_liq']:.4f} J/(mol·K), "
              f"T_fusion = {data['T_fusion']} K, polymorphs = {len(data['polymorphs'])}")

        phases = [p for p in bl.phases if not any(s.lower() in p['name'].lower() for s in ALL_SS_PHASES)]

        eqs = build_thermodynamic_expressions(
            param_format=bl._param_format,
            ga_expr=bl.component_data[bl.components[0]]['H_liq'] - \
                t_sym * bl.component_data[bl.components[0]]['S_liq'],
            gb_expr=bl.component_data[bl.components[1]]['H_liq'] - \
                t_sym * bl.component_data[bl.components[1]]['S_liq'])
        
        hull_points = np.array([[0, 0]] + [[p['comp'], p['enthalpy']] for p in phases if 'comp' in p] + [[1, 0]])
        eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])

        kwargs.update({
            'sys_name': bl.sys_name,
            'components': bl.components,
            'component_data': bl.component_data,
            'mpds_json': bl.mpds_json,
            'digitized_liq': bl.digitized_liq,
            'temp_range': bl.temp_range,
            'dft_type': bl.dft_type,
            'dft_ch': bl.dft_ch,
            'phases': phases,
            'params': bl._params,
            'param_format': bl._param_format,
            'eqs': eqs,
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
        s_offsets = ss_model["deltaS_a_jmol_k"] * xa + ss_model["deltaS_b_jmol_k"] * x_arr
        s_total = s_offsets + s_conf

        h_offsets = ss_model["deltaH_a_jmol"] * xa + ss_model["deltaH_b_jmol"] * x_arr
        h_mix = ss_model["omega_jmol"] * xa * x_arr
        h_total = h_offsets + h_mix

        return h_total, s_total
    
    def solid_solution_gibbs(self, ss_name: str, x_vals: np.ndarray, temp_k: float) -> np.ndarray:
        h_vals, s_vals = self.solid_solution_h_s(ss_name, x_vals=x_vals)
        return h_vals - float(temp_k) * s_vals

    def to_HSX(self, fmt: str = "dict") -> dict | pd.DataFrame: 

        data = super().to_HSX(fmt="dict")


        for ss_name in self.ss_names:
            ss_h_vals, ss_s_vals = self.solid_solution_h_s(ss_name)
            data["X"].extend([float(x) for x in _x_vals])
            data["S"].extend([float(s) for s in ss_s_vals])
            data["H"].extend([float(h) for h in ss_h_vals])
            data["Phase Name"].extend([ss_name] * len(_x_vals))

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            return pd.DataFrame(data)
        raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")

    def update_phase_points(self) -> dict:
        """
        Calculates the phase points for given parameter values using the HSX class.

        This method converts phase data into the HSX form and uses HSX code to calculate the liquidus
        and low-temperature DFT phase boundaries.

        Returns:
            data (dict): A dictionary containing the phase data in HSX format, including phase names and components.
        """
        data = self.to_HSX()
        hsx_dict = {
            'data': data,
            'phases': [phase['name'] for phase in self.phases],
            'comps': self.components
        }
        self.hsx = HSX(hsx_dict, [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15], use_filter_2=False)
        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]
        return data


# def _linear_ref_expr(enthalpy_jmol: float, transition_temp_k: float) -> sp.Expr:
#     if enthalpy_jmol == 0 or transition_temp_k == 0:
#         return sp.Integer(0)
#     return float(enthalpy_jmol) - t_sym * float(enthalpy_jmol) / float(transition_temp_k)



def auto_chg_temps_k(system: SolidSolutionBinaryLiquid) -> list[float]:
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    solid_phases = [
        phase
        for phase in system.phases
        if phase["name"] not in system.components + ["L"] and len(phase["points"]) > 0
    ]

    def _finite_temps(points: list) -> list[float]:
        vals: list[float] = []
        for point in points:
            if len(point) < 2:
                continue
            t_val = float(point[1])
            if np.isfinite(t_val):
                vals.append(t_val)
        return vals

    t_candidates: list[float] = []
    if solid_phases:
        for phase in solid_phases:
            t_candidates.extend(_finite_temps(phase["points"]))
    if not t_candidates and system.phases[-1]["points"]:
        t_candidates.extend(_finite_temps(system.phases[-1]["points"]))
    if not t_candidates:
        t_candidates.extend(
            float(component[1])
            for component in system.component_data.values()
            if len(component) > 1 and np.isfinite(float(component[1])) and float(component[1]) > 0
        )

    t_high = max(t_candidates) if t_candidates else 2000.0
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
    finite_t_vals = [float(t) for t in t_vals_k if np.isfinite(float(t))]
    finite_t_vals = sorted(set(finite_t_vals))
    for temp_k in reversed(finite_t_vals):
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
        return f"{phase_label} ({system.components[0]}, {system.components[1]})"
    return phase_label


def _phase_color_map(system: SolidSolutionBinaryLiquid) -> dict[str, str]:
    cached = getattr(system, "_ss_phase_color_map", None)
    if cached is not None:
        return cached

    reserved = set(SS_FIXED_COLORS.values())
    reserved.add("cornflowerblue")
    base_palette = [c for c in px.colors.qualitative.Pastel if c not in reserved]
    if not base_palette:
        base_palette = px.colors.qualitative.Pastel

    phase_map: dict[str, str] = {"L": "cornflowerblue"}

    # Mirror original HSX behavior: line-compound phases are color-coded from a palette.
    line_phases = [
        p for p in system.hsx.phases
        if p != "L" and p not in system.ss_names
    ]
    for idx, phase in enumerate(line_phases):
        phase_map[phase] = base_palette[idx % len(base_palette)]

    # Add deterministic, unique colors for solid-solution phases.
    fallback_ss_palette = ["#6c5ce7", "#00a896", "#ef476f", "#ffd166"]
    used_colors = set(phase_map.values())
    fallback_idx = 0
    for ss_name in system.ss_names:
        fixed = SS_FIXED_COLORS.get(ss_name)
        if fixed is not None and fixed not in used_colors:
            phase_map[ss_name] = fixed
            used_colors.add(fixed)
            continue
        while fallback_idx < len(fallback_ss_palette) and fallback_ss_palette[fallback_idx] in used_colors:
            fallback_idx += 1
        if fallback_idx < len(fallback_ss_palette):
            phase_map[ss_name] = fallback_ss_palette[fallback_idx]
            used_colors.add(fallback_ss_palette[fallback_idx])
            fallback_idx += 1
        else:
            phase_map[ss_name] = "#3a86ff"

    system._ss_phase_color_map = phase_map
    return phase_map


def _phase_color(system: SolidSolutionBinaryLiquid, phase_label: str) -> str:
    return _phase_color_map(system).get(phase_label, "#555555")


def build_tx_scatter_with_solid_solution(
    system: SolidSolutionBinaryLiquid,
    include_digitized_liquidus: bool = True,
) -> go.Figure:
    """Diagnostic TX scatter using raw points from HSX compute_tx (no envelope post-processing)."""
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    df_tx, _, _, _ = system.hsx.compute_tx()
    df = df_tx.copy()
    df["x_at"] = df["x"].astype(float) * 100.0
    df["t_c"] = df["t"].astype(float) - 273.15
    df["label_display"] = [
        _phase_display_name(system, str(label)) for label in df["label"]
    ]

    color_map = {
        _phase_display_name(system, phase): _phase_color(system, phase)
        for phase in system.hsx.phases
    }

    fig = px.scatter(
        df,
        x="x_at",
        y="t_c",
        color="label_display",
        color_discrete_map=color_map,
        title=f"{system.sys_name} TX Scatter (Raw HSX compute_tx Points)",
        width=920,
        height=700,
    )

    fig.update_traces(marker={"size": 7, "opacity": 0.85})

    if include_digitized_liquidus and system.digitized_liq:
        fig.add_trace(
            go.Scatter(
                x=[float(point[0] * 100.0) for point in system.digitized_liq],
                y=[float(point[1] - 273.15) for point in system.digitized_liq],
                mode="lines",
                line={"color": "#b82e2e", "width": 2.0, "dash": "dash"},
                name="Assessed Liquidus",
            )
        )

    y_lo = float(system.temp_range[0] - 273.15)
    y_hi = float(system.temp_range[-1] - 273.15) + 100.0
    fig.update_layout(
        xaxis={"range": [0, 100], "title": f"X_{system.components[1]} (at. %)"},
        yaxis={"range": [y_lo, y_hi], "title": "T [C]"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        font={"size": 18},
    )
    fig.update_xaxes(
        mirror=True,
        ticks="inside",
        showline=True,
        linecolor="gray",
        linewidth=2,
        tickcolor="gray",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="inside",
        showline=True,
        linecolor="gray",
        linewidth=2,
        tickcolor="gray",
    )

    return fig


def build_tx_with_solid_solution(
    system: SolidSolutionBinaryLiquid,
    show_tie_lines: bool = True,
    tie_line_stride: int = 1,
) -> go.Figure:
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    df_tx, final_phases, simplices, temps_k = system.hsx.compute_tx()
    valid_mask = np.isfinite(temps_k) & (temps_k >= 0.0)
    if not np.all(valid_mask):
        temps_k = temps_k[valid_mask]
        simplices = simplices[valid_mask]
        final_phases = final_phases[valid_mask]

    if len(df_tx) == 0:
        raise RuntimeError("HSX compute_tx produced no points for line rendering.")

    if "t" in df_tx.columns:
        df_tx = df_tx[np.isfinite(df_tx["t"].astype(float)) & (df_tx["t"].astype(float) >= 0.0)].copy()
    if len(df_tx) == 0 or len(temps_k) == 0:
        raise RuntimeError("No physically valid (T >= 0 K) HSX coexistence points available for line rendering.")

    temps_c = temps_k - 273.15
    df = df_tx.copy()
    df["x_at"] = df["x"].astype(float) * 100.0
    df["t_c"] = df["t"].astype(float) - 273.15

    def _reduce_simplex_vertices(comps: list[float], phases: list[str]) -> tuple[list[float], list[str]]:
        """Reduce duplicate phase vertices in a simplex to one representative point per phase.

        For duplicate phase vertices (A, A, B), keep the A-point closest in composition
        to the non-A vertex. This mirrors the original plot_tx duplicate-pruning behavior
        and generalizes it to SS phases.
        """
        order = np.argsort(np.asarray(comps, dtype=float))
        comps_sorted = [float(np.asarray(comps, dtype=float)[i]) for i in order]
        phases_sorted = [str(np.asarray(phases, dtype=object)[i]) for i in order]

        unique_in_order: list[str] = []
        for p in phases_sorted:
            if p not in unique_in_order:
                unique_in_order.append(p)

        reduced: list[tuple[float, str]] = []
        for phase in unique_in_order:
            idxs = [i for i, p in enumerate(phases_sorted) if p == phase]
            if len(idxs) == 1:
                keep_idx = idxs[0]
            elif len(idxs) == 2:
                other = [i for i in range(len(comps_sorted)) if i not in idxs]
                if other:
                    other_comp = float(np.mean([comps_sorted[i] for i in other]))
                    keep_idx = min(idxs, key=lambda i: abs(comps_sorted[i] - other_comp))
                else:
                    keep_idx = idxs[0]
            else:
                keep_idx = idxs[len(idxs) // 2]
            reduced.append((comps_sorted[keep_idx], phase))

        reduced.sort(key=lambda item: item[0])
        return [x for x, _ in reduced], [p for _, p in reduced]

    # Build simplex-level reduced TX points similar to original plot_tx combined_list/new_tx flow.
    reduced_rows: list[list[float | str]] = []
    for i, simplex in enumerate(simplices):
        comps = [float(system.hsx.points[v][0]) for v in simplex]
        phases = [str(final_phases[i][j]) for j in range(len(simplex))]
        x_red, p_red = _reduce_simplex_vertices(comps, phases)
        t_c = float(temps_c[i])
        for x_val, phase in zip(x_red, p_red):
            reduced_rows.append([x_val * 100.0, t_c, phase, _phase_color(system, phase)])

    reduced_df = pd.DataFrame(reduced_rows, columns=["x", "t", "label", "color"])

    all_t_candidates = [float(np.min(df["t_c"])), float(np.max(df["t_c"]))]
    if system.digitized_liq:
        liq_t = [float(point[1] - 273.15) for point in system.digitized_liq]
        all_t_candidates.extend([min(liq_t), max(liq_t)])
    y_lo = min(float(system.temp_range[0] - 273.15), min(all_t_candidates))
    y_hi = max(float(system.temp_range[1] - 273.15), max(all_t_candidates))
    y_pad = max(40.0, 0.04 * (y_hi - y_lo))
    y_lo -= y_pad
    y_hi += y_pad
    y_floor = max(-273.15, y_lo)

    fig = go.Figure()

    # Liquidus selection mirrors original plot_tx logic: one liquid point per composition.
    liq_df = (
        df[df["label"] == "L"][["x_at", "t_c"]]
        .sort_values(["x_at", "t_c"])
        .drop_duplicates(subset="x_at", keep="first")
    )
    if not liq_df.empty:
        liq_x = liq_df["x_at"].to_numpy(dtype=float)
        liq_t = liq_df["t_c"].to_numpy(dtype=float)
        for seg_x, seg_y in _split_segments(liq_x, liq_t):
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    line={"color": _phase_color(system, "L"), "width": 2.8},
                    name="L",
                    showlegend=False,
                )
            )

    phase_order = [p for p in system.hsx.phases if p != "L"]
    for phase in phase_order:
        phase_df = reduced_df[reduced_df["label"] == phase]
        if phase_df.empty:
            continue

        color = _phase_color(system, phase)
        name = _phase_display_name(system, phase)

        x_unique = np.array(sorted(phase_df["x"].unique()), dtype=float)
        t_min = np.array([phase_df.loc[phase_df["x"] == x, "t"].min() for x in x_unique], dtype=float)
        t_max = np.array([phase_df.loc[phase_df["x"] == x, "t"].max() for x in x_unique], dtype=float)

        x_upper_plot = x_unique.copy()
        t_upper_plot = t_max.copy()
        x_lower_plot = x_unique.copy()
        t_lower_plot = t_min.copy()

        # For SS phases, extend upper/lower branches to a single extrapolated junction
        # (instead of adding vertical endpoint connectors that create a closed polygon).
        spread = t_max - t_min
        if phase in system.ss_names and x_unique.size >= 3 and np.any(spread > 1e-6):
            side = "left" if spread[0] <= spread[-1] else "right"

            def _side_line(xv: np.ndarray, yv: np.ndarray, which: str) -> tuple[float, float] | None:
                if which == "left":
                    x1, y1 = float(xv[0]), float(yv[0])
                    x2, y2 = float(xv[1]), float(yv[1])
                else:
                    x1, y1 = float(xv[-2]), float(yv[-2])
                    x2, y2 = float(xv[-1]), float(yv[-1])
                dx = x2 - x1
                if np.isclose(dx, 0.0, atol=1e-12):
                    return None
                m = (y2 - y1) / dx
                b = y1 - m * x1
                return m, b

            up_line = _side_line(x_upper_plot, t_upper_plot, side)
            lo_line = _side_line(x_lower_plot, t_lower_plot, side)
            if up_line is not None and lo_line is not None:
                m_up, b_up = up_line
                m_lo, b_lo = lo_line
                if not np.isclose(m_up, m_lo, atol=1e-12):
                    x_int = (b_lo - b_up) / (m_up - m_lo)
                    t_int = m_up * x_int + b_up

                    ext_ok = (
                        np.isfinite(x_int)
                        and np.isfinite(t_int)
                        and (x_int < x_unique[0] if side == "left" else x_int > x_unique[-1])
                        and (abs(x_int - (x_unique[0] if side == "left" else x_unique[-1])) <= 20.0)
                        and (y_floor - 300.0 <= t_int <= y_hi + 300.0)
                    )

                    if ext_ok:
                        if side == "left":
                            x_upper_plot = np.insert(x_upper_plot, 0, x_int)
                            t_upper_plot = np.insert(t_upper_plot, 0, t_int)
                            x_lower_plot = np.insert(x_lower_plot, 0, x_int)
                            t_lower_plot = np.insert(t_lower_plot, 0, t_int)
                        else:
                            x_upper_plot = np.append(x_upper_plot, x_int)
                            t_upper_plot = np.append(t_upper_plot, t_int)
                            x_lower_plot = np.append(x_lower_plot, x_int)
                            t_lower_plot = np.append(t_lower_plot, t_int)

        if x_unique.size == 1:
            x0 = float(x_unique[0])
            y0 = float(t_min[0])
            y1 = float(t_max[0])
            if phase in system.components and (x0 < 0.5 or x0 > 99.5):
                y0 = y_lo
            elif phase not in system.ss_names and phase not in system.components:
                # Match original plot_tx behavior: extend line compounds to low temperature.
                y0 = y_floor
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0],
                    y=[y0, y1],
                    mode="lines",
                    line={"color": color, "width": 2.4},
                    name=name,
                    showlegend=True,
                )
            )
            continue

        show_legend = True
        for seg_x, seg_y in _split_segments(x_upper_plot, t_upper_plot):
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    line={"color": color, "width": 2.4},
                    name=name,
                    showlegend=show_legend,
                )
            )
            show_legend = False

        # Draw secondary branch where the phase occupies a temperature interval at fixed composition.
        spread = t_upper_plot - t_lower_plot
        if np.any(spread > 1e-6):
            lower_segments = _split_segments(x_lower_plot, t_lower_plot)
            for seg_x, seg_y in lower_segments:
                fig.add_trace(
                    go.Scatter(
                        x=seg_x,
                        y=seg_y,
                        mode="lines",
                        line={"color": color, "width": 1.2, "dash": "dot"},
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

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

    # Draw only three-phase liquidus invariant tie-lines (original plot_tx style).
    if show_tie_lines:
        try:
            inv_points, _, _ = system.hsx.liquidus_invariants()
        except Exception:
            inv_points = {}

        for inv_type in ("Eutectics", "Peritectics", "Misc Gaps"):
            for temp_c, _, comps, phases in inv_points.get(inv_type, []):
                if not np.isfinite(float(temp_c)) or float(temp_c) < -273.15:
                    continue
                if len(comps) != 3:
                    continue
                phase_labels = [str(p) for p in phases]
                if "L" not in phase_labels:
                    continue
                if inv_type == "Misc Gaps":
                    non_liq = [p for p in phase_labels if p != "L"]
                    if non_liq and len(set(non_liq)) <= 1 and any(p in system.ss_names for p in non_liq):
                        continue

                comps_pct = [float(x) * 100.0 for x in comps]
                fig.add_trace(
                    go.Scatter(
                        x=comps_pct,
                        y=[float(temp_c)] * 3,
                        mode="lines",
                        line={"color": "Silver", "width": 1.2},
                        showlegend=False,
                        hovertemplate=f"{inv_type[:-1]} invariant<br>T: %{{y:.1f}} C<extra></extra>",
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
                text=_phase_display_name(system, ss_name),
                showarrow=False,
                font={"size": 14, "color": "black"},
                textangle=-90,
            )

        # ss_color = _phase_color(system, ss_name)

        # if all(isinstance(v, list) for v in system.component_data.values()):
        #     ss_ref_endpoints = [
        #     (0.0, float(system.component_data[system.components[0]][0]) - 273.15),
        #     (100.0, float(system.component_data[system.components[1]][0]) - 273.15),
        # ]
        # elif all(isinstance(v, dict) for v in system.component_data.values()):
        #     ss_ref_endpoints = [
        #     (0.0, float(system.component_data[system.components[0]]['T_fusion']) - 273.15),
        #     (100.0, float(system.component_data[system.components[1]]['T_fusion']) - 273.15),
        # ]
        # else:
        #     raise ValueError("component_data values must be either a list of dicts (legacy format) or dict of dicts (element-db format).")

        # fig.add_trace(
        #     go.Scatter(
        #         x=[point[0] for point in ss_ref_endpoints],
        #         y=[point[1] for point in ss_ref_endpoints],
        #         mode="markers",
        #         marker={"color": ss_color, "size": 8},
        #         name=f"{ss_name} endpoint refs",
        #         showlegend=False,
        #     )
        # )

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


def plot_hsx_with_solid_solution_blocks(
    system: SolidSolutionBinaryLiquid,
    show_hull_simplices: bool = True,
    simplex_color: str = "cyan",
    simplex_opacity: float = 0.28,
    ss_block_opacity: float = 0.30,
) -> go.Figure:
    """HSX diagnostic plot with solid-solution phases rendered as phase-level blocks.

    This mirrors the original HSX plot style (scatter + lower-hull simplices) and adds
    one mesh block per solid-solution phase so each continuous solution is visually grouped.
    """
    if not system.phases[-1]["points"]:
        system.update_phase_points()

    hsx_obj = system.hsx
    simplices = hsx_obj.hull()
    df = hsx_obj.df
    points = hsx_obj.points
    # ss_set = set(system.ss_names)
    scatter_colors = [_phase_color(system, str(phase)) for phase in df["Phase"]]

    fig = go.Figure()

    # Base scatter for all HSX points.
    fig.add_trace(
        go.Scatter3d(
            x=df["X [Fraction]"],
            y=df["S [J/mol/K]"],
            z=df["H [J/mol]"],
            mode="markers",
            marker={"size": 4, "opacity": 0.55, "color": scatter_colors},
            name="HSX points",
            showlegend=False,
            hovertemplate=(
                "Phase: %{customdata}<br>"
                "X: %{x:.4f}<br>"
                "S: %{y:.4f}<br>"
                "H: %{z:.4f}<extra></extra>"
            ),
            customdata=df["Phase"],
        )
    )

    # Overlay lower-hull simplices used for TX construction.
    if show_hull_simplices:
        for simplex in simplices:
            x_coords = points[simplex, 0]
            y_coords = points[simplex, 1]
            z_coords = points[simplex, 2]
            fig.add_trace(
                go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    i=np.array([0]),
                    j=np.array([1]),
                    k=np.array([2]),
                    opacity=float(simplex_opacity),
                    color=simplex_color,
                    name="Hull simplex",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Render each SS phase as a single colored mesh block.
    for ss_name in system.ss_names:
        ss_df = df[df["Phase"] == ss_name]
        if ss_df.empty:
            continue

        ss_color = _phase_color(system, ss_name)
        ss_x = ss_df["X [Fraction]"].to_numpy(dtype=float)
        ss_y = ss_df["S [J/mol/K]"].to_numpy(dtype=float)
        ss_z = ss_df["H [J/mol]"].to_numpy(dtype=float)

        fig.add_trace(
            go.Mesh3d(
                x=ss_x,
                y=ss_y,
                z=ss_z,
                alphahull=5,
                opacity=float(ss_block_opacity),
                color=ss_color,
                name=f"{ss_name} block",
                hovertemplate=(
                    f"{ss_name} block<br>"
                    "X: %{x:.4f}<br>"
                    "S: %{y:.4f}<br>"
                    "H: %{z:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Create clean legend entries for all non-liquid phases and liquid.
    legend_labels = [p for p in hsx_obj.phases if p != "L"] + ["L"]
    for label in legend_labels:
        color = _phase_color(system, label)
        legend_name = _phase_display_name(system, label)
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 8, "color": color},
                name=legend_name,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"<b>{system.sys_name} HSX Convex Hull Debug (Solid-Solution Blocks)</b>",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "S [scaled J/mol/K]",
            "zaxis_title": "H [scaled J/mol]",
        },
        legend={"itemsizing": "constant", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 0.02},
        font={"size": 14},
        width=980,
        height=760,
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
                "omega_jmol": float(ss_model["omega_jmol"]),
                "deltaH_a_jmol": float(ss_model["deltaH_a_jmol"]),
                "deltaH_b_jmol": float(ss_model["deltaH_b_jmol"]),
                "deltaS_a_jmol_k": float(ss_model["deltaS_a_jmol_k"]),
                "deltaS_b_jmol_k": float(ss_model["deltaS_b_jmol_k"]),
                "refs": ss_model["refs"],
            }
            for ss_name, ss_model in system.ss_models.items()
        },
    }


def main():

    # elts = ["Zr", "Hf", "Nb", "W", "Al"]
    # system_combos = [sorted(elts[i], elts[j]) for i in range(len(elts)) for j in range(i + 1, len(elts))]
   
    # ref modes: omegas-legacy, binary-cache, element-db
    # sys_name = "Nb-Zr"
    sys_name = "Nb-W"
    # sys_name = "Al-Zr"
    # sys_name = "Te-Zr"
    system = SolidSolutionBinaryLiquid.from_cache(sys_name, pd_ind=0, ref_mode='omegas-legacy', param_format='comb-exp',
                                                  omegas_path=cfg.data_dir / "omegas_hcp.json")
    # system.update_phase_points()
    # plot_hsx_with_solid_solution_blocks(system).show()

    print("Solid-solution models:")
    for ss_name, ss_model in system.ss_models.items():
        print(
            f"  phase={ss_name}",
            f"omega={ss_model['omega_jmol']:.6f} J/mol",
            f"deltaH=({ss_model['deltaH_a_jmol']:.6f}, {ss_model['deltaH_b_jmol']:.6f}) J/mol",
            f"deltaS=({ss_model['deltaS_a_jmol_k']:.6f}, {ss_model['deltaS_b_jmol_k']:.6f}) J/(mol*K)",
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
                f"deltaH={ref['delta_h_jmol']:.6f} J/mol | "
                f"deltaS={ref['delta_s_jmol_k']:.6f} J/(mol*K)"
            )
        print()

    # plot_hsx_with_solid_solution_blocks(system).show()

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
    fit_fig.show()

    tx_scatter_fig = build_tx_scatter_with_solid_solution(system)
    tx_scatter_fig.show()

    hsx_debug_fig = plot_hsx_with_solid_solution_blocks(system)
    hsx_debug_fig.show()

    t_vals_k = auto_chg_temps_k(system)
    chg_fig = build_chg_with_solid_solution(system, t_vals_k=t_vals_k)
    chg_fig.show()
    chg_fig_hidden = build_chg_with_solid_solution(system, t_vals_k=t_vals_k, hide_unstable_solution_curves=True)
    chg_fig_hidden.show()


    # fit_path = cfg.project_root / "figures" / f"{sys_name}_fit_plus_liq.html"
    # fit_fig.write_html(str(fit_path), include_plotlyjs="cdn")
    # print(f"Saved plot: {fit_path}")

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
