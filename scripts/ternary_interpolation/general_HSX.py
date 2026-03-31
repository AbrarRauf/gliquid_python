'''
Author: Abrar Rauf, Joshua Wilwerth
This module contains the classes for general interpolation and general phase diagram plotting.
'''
import os
import time
import sys
import json
import numpy as np
np.set_printoptions(legacy='1.25')
import pandas as pd
import sympy as sp
import plotly.express as px
import plotly.graph_objects as go

from typing import List
from emmet.core.utils import jsanitize
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Element, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from scipy.spatial import Delaunay, QhullError, cKDTree
from copy import deepcopy
from pathlib import Path

WORKSPACE_ROOT = Path("C:\\Users\\AbrarRauf\\University of Michigan Dropbox\\Abrar Rauf\\WHSun_Lab\\G_liquid")

from gliquid.config import phase_transitions_file
from gliquid.binary import (
    BinaryLiquid,
    linear_expr, exponential_expr, combined_expr)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # If importing this file into a script from a different dir
from extensive_hull_main import gliq_lowerhull3, gen_hyperplane_eqns2_optimized
import random
from auth import mpapi_key

mpr = MPRester(mpapi_key)  

R = 8.314  # J/(mol*K)
EV_PER_ATOM_TO_J_PER_MOL = 96485.33212

REF_SS_PHASES = ("BCC", "FCC", "HCP")
REF_SS_SPACEGROUPS = {
    "BCC": 229,
    "FCC": 225,
    "HCP": 194,
}
REF_SS_PHASE_NAME_MARKERS = {
    "BCC": ("bcc", "im-3m"),
    "FCC": ("fcc", "fm-3m"),
    "HCP": ("hcp", "p6_3/mmc"),
}
DEFAULT_REF_SS_OMEGAS_PATH = WORKSPACE_ROOT / "matrix_data" / "omegas.json"


# ==============================================================================
# SYMBOLIC VARIABLE MANAGEMENT
# ==============================================================================

from typing import Dict, Tuple, Optional, Callable, Any
from itertools import combinations


# ==============================================================================
# PARAMETER EXPRESSION TEMPLATES (Module-level)
# ==============================================================================

def _linear_expr(a: sp.Expr, b: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """Linear temperature dependence: L = a + b*T"""
    return a + b * t


def _exponential_expr(a: sp.Expr, b: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """Exponential temperature dependence: L = a * exp(-T/b)"""
    return a * sp.exp(-t / b)


def _combined_expr(a: sp.Expr, b: sp.Expr, t: sp.Symbol, tau: sp.Expr = sp.Integer(8000)) -> sp.Expr:
    """Combined linear-exponential: L = (a + b*T) * exp(-T/tau)"""
    return (a + b * t) * sp.exp(-t / tau)


PARAM_EXPR_REGISTRY: Dict[str, Callable] = {
    'linear': _linear_expr,
    'exponential': _exponential_expr,
    'combined': _combined_expr,
}


# ==============================================================================
# INTERPOLATION SCHEMES (Module-level)
# ==============================================================================

def _muggianu_weights(x: Tuple[sp.Expr, ...], i: int, j: int) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Muggianu interpolation: symmetric projection onto binary edges."""
    xi, xj = x[i], x[j]
    remainder = 1 - xi - xj
    xi_eff = xi + remainder / 2
    xj_eff = xj + remainder / 2
    return xi * xj, xi_eff, xj_eff


def _kohler_weights(x: Tuple[sp.Expr, ...], i: int, j: int) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Kohler interpolation: normalized projection."""
    xi, xj = x[i], x[j]
    sum_ij = xi + xj
    xi_eff = sp.Piecewise((sp.Rational(1, 2), sp.Eq(sum_ij, 0)), (xi / sum_ij, True))
    xj_eff = sp.Piecewise((sp.Rational(1, 2), sp.Eq(sum_ij, 0)), (xj / sum_ij, True))
    return xi * xj, xi_eff, xj_eff


def _linear_weights(x: Tuple[sp.Expr, ...], i: int, j: int) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Linear interpolation: no transformation."""
    return x[i] * x[j], x[i], x[j]


INTERPOLATION_SCHEMES: Dict[str, Callable] = {
    'linear': _linear_weights,
    'muggianu': _muggianu_weights,
    'kohler': _kohler_weights,
}


# ==============================================================================
# THERMODYNAMIC EXPRESSION BUILDER CLASS
# ==============================================================================

class ThermoExprBuilder:
    """
    Class for building thermodynamic expressions for multi-component systems.
    
    Encapsulates symbolic variable management, expression building, and
    convenience methods for creating reference Gibbs energies and L parameters.
    
    Attributes:
        n_components: Number of components in the system.
        param_format: Format for L parameter temperature dependence ('linear', 'exponential', 'combined').
        interp_scheme: Interpolation scheme ('linear', 'muggianu', 'kohler').
        t: Temperature symbol.
        x: Tuple of independent atom-fraction symbols.
        atom_fractions: Tuple of all atom fractions (including dependent one).
        binary_pairs: List of (i, j) tuples for all binary pairs.
        
        # Expression outputs (populated after build())
        g_ref: Reference Gibbs energy expression.
        g_ideal: Ideal mixing Gibbs energy expression.
        g_excess: Excess Gibbs energy expression.
        g_liquid: Total liquid Gibbs energy expression.
        s_liquid: Entropy expression.
        h_liquid: Enthalpy expression.
    
    Example:
        >>> builder = ThermodynamicExpressionBuilder(n_components=3)
        >>> builder.set_reference_gibbs_from_fusion([10000, 12000, 8000], [800, 900, 700])
        >>> builder.set_binary_L_params({(0,1): [5000, -2, -3000, 1.5], ...})
        >>> builder.build()
        >>> G = builder.g_liquid  # Access the built expression
    """
    
    def __init__(
        self,
        n_components: int,
        param_format: str = 'linear',
        interp_scheme: str = 'linear',
        tau: float = 8000
    ):
        """
        Initialize the expression builder for an n-component system.
        
        Args:
            n_components: Number of components (must be >= 2).
            param_format: Format for L parameters ('linear', 'exponential', 'combined').
            interp_scheme: Interpolation scheme ('linear', 'muggianu', 'kohler').
            tau: Time constant for combined expression format (default 8000).
        """
        if n_components < 2:
            raise ValueError("System must have at least 2 components.")
        
        self.n_components = n_components
        self.param_format = param_format
        self.interp_scheme = interp_scheme
        self.tau = sp.Integer(tau)
        
        # Validate inputs
        if param_format not in PARAM_EXPR_REGISTRY:
            raise ValueError(f"Unknown param_format '{param_format}'. Available: {list(PARAM_EXPR_REGISTRY.keys())}")
        if interp_scheme not in INTERPOLATION_SCHEMES:
            raise ValueError(f"Unknown interp_scheme '{interp_scheme}'. Available: {list(INTERPOLATION_SCHEMES.keys())}")
        
        # Initialize symbolic variables
        self._init_symbols()
        
        # Input storage (set via methods or directly)
        self.g_ref_exprs: Optional[Tuple[sp.Expr, ...]] = None
        self.binary_L_exprs: Optional[Dict[Tuple[int, int], Tuple[sp.Expr, sp.Expr]]] = None
        self.binary_weights: Optional[Dict[Tuple[int, int], sp.Expr]] = None
        self.higher_order_exprs: Dict[Tuple[int, ...], sp.Expr] = {}
        
        # Output expressions (populated after build())
        self.g_ref: Optional[sp.Expr] = None
        self.g_ideal: Optional[sp.Expr] = None
        self.g_excess: Optional[sp.Expr] = None
        self.g_liquid: Optional[sp.Expr] = None
        self.s_liquid: Optional[sp.Expr] = None
        self.h_liquid: Optional[sp.Expr] = None
    
    def _init_symbols(self):
        """Initialize symbolic variables for the system."""
        # Temperature symbol
        self.t = sp.Symbol('T', real=True, positive=True)

        # Independent atom fractions correspond to the latter components
        # in canonical sorted order: x1, x2, ..., x[n-1].
        self.x = sp.symbols(f'x1:{self.n_components}', real=True, nonnegative=True)
        if self.n_components == 2:
            self.x = (self.x,) if not isinstance(self.x, tuple) else self.x

        # Component-0 atom fraction is dependent by closure.
        x_indep = list(self.x)
        x_dep = 1 - sum(self.x)
        self.atom_fractions = tuple([x_dep] + x_indep)
        
        # Binary pairs
        self.binary_pairs = list(combinations(range(self.n_components), 2))
    
    @property
    def n_binary_pairs(self) -> int:
        """Number of binary pairs: n*(n-1)/2"""
        return len(self.binary_pairs)
    
    def get_atom_fraction(self, index: int) -> sp.Expr:
        """Get atom-fraction expression for component at given index."""
        if index < 0 or index >= self.n_components:
            raise IndexError(f"Component index {index} out of range [0, {self.n_components - 1}]")
        return self.atom_fractions[index]
    
    # =========================================================================
    # Input setters
    # =========================================================================
    
    def set_reference_gibbs(self, g_ref_exprs: Tuple[sp.Expr, ...]):
        """
        Set reference Gibbs energy expressions directly.
        
        Args:
            g_ref_exprs: Tuple of sympy expressions for each component's G_ref.
        """
        if len(g_ref_exprs) != self.n_components:
            raise ValueError(f"Expected {self.n_components} reference energies, got {len(g_ref_exprs)}")
        self.g_ref_exprs = tuple(g_ref_exprs)
    
    def set_reference_gibbs_from_fusion(
        self,
        fusion_enthalpies: List[float],
        fusion_temps: List[float]
    ):
        """
        Set reference Gibbs energies from fusion data.
        
        G_ref_i = H_fus_i - T * S_fus_i, where S_fus = H_fus / T_fus.
        
        Args:
            fusion_enthalpies: List of fusion enthalpies [J/mol] for each component.
            fusion_temps: List of fusion temperatures [K] for each component.
        """
        if len(fusion_enthalpies) != self.n_components:
            raise ValueError(f"Expected {self.n_components} enthalpies, got {len(fusion_enthalpies)}")
        if len(fusion_temps) != self.n_components:
            raise ValueError(f"Expected {self.n_components} temperatures, got {len(fusion_temps)}")
        
        g_refs = []
        for h_fus, t_fus in zip(fusion_enthalpies, fusion_temps):
            s_fus = h_fus / t_fus
            g_ref = h_fus - self.t * s_fus
            g_refs.append(g_ref)
        
        self.g_ref_exprs = tuple(g_refs)
    
    def set_binary_L_exprs(self, binary_L_exprs: Dict[Tuple[int, int], Tuple[sp.Expr, sp.Expr]]):
        """
        Set binary L expressions directly.
        
        Args:
            binary_L_exprs: Dict mapping (i, j) to (L0_expr, L1_expr).
        """
        self.binary_L_exprs = binary_L_exprs
    
    def set_binary_L_params(self, binary_params: Dict[Tuple[int, int], List[float]]):
        """
        Set binary L expressions from numeric parameters.
        
        Args:
            binary_params: Dict mapping (i, j) to [L0_a, L0_b, L1_a, L1_b].
                For 'linear': L = a + b*T
                For 'exponential': L = a * exp(-T/b)
                For 'combined': L = (a + b*T) * exp(-T/tau)
        """
        expr_func = PARAM_EXPR_REGISTRY[self.param_format]
        
        result = {}
        for pair, params in binary_params.items():
            if self.param_format == 'combined':
                l0 = _combined_expr(params[0], params[1], self.t, self.tau)
                l1 = _combined_expr(params[2], params[3], self.t, self.tau)
            else:
                l0 = expr_func(params[0], params[1], self.t)
                l1 = expr_func(params[2], params[3], self.t)
            result[pair] = (l0, l1)
        
        self.binary_L_exprs = result
    
    def set_binary_weights(self, binary_weights: Dict[Tuple[int, int], sp.Expr]):
        """Set optional weight factors for each binary pair."""
        self.binary_weights = binary_weights
    
    def add_higher_order_term(self, indices: Tuple[int, ...], expr: sp.Expr):
        """
        Add a higher-order interaction term.
        
        Args:
            indices: Tuple of component indices involved (e.g., (0, 1, 2) for ternary).
            expr: Sympy expression for the interaction parameter.
        """
        self.higher_order_exprs[indices] = expr
    
    # =========================================================================
    # Expression building
    # =========================================================================
    
    def _build_ideal_mixing_gibbs(self) -> sp.Expr:
        """Build ideal mixing Gibbs energy: G_ideal = R*T * sum(xi * ln(xi))."""
        mixing_terms = []
        for xi in self.atom_fractions:
            term = sp.Piecewise(
                (0, sp.Eq(xi, 0)),
                (xi * sp.log(xi), True)
            )
            mixing_terms.append(term)
        return R * self.t * sum(mixing_terms)
    
    def _build_reference_gibbs(self) -> sp.Expr:
        """Build reference Gibbs energy: G_ref = sum(xi * Gi_ref)."""
        return sum(
            self.atom_fractions[i] * self.g_ref_exprs[i]
            for i in range(self.n_components)
        )
    
    def _build_binary_excess_gibbs(
        self,
        i: int,
        j: int,
        l0_expr: sp.Expr,
        l1_expr: sp.Expr,
        weight_factor: sp.Expr = sp.Integer(1)
    ) -> sp.Expr:
        """Build excess Gibbs contribution from one binary pair."""
        weight_func = INTERPOLATION_SCHEMES[self.interp_scheme]
        weight, xi_eff, xj_eff = weight_func(self.atom_fractions, i, j)
        
        # Redlich-Kister 2-term polynomial
        rk_expr = l0_expr + l1_expr * (xi_eff - xj_eff)
        
        return weight * weight_factor * rk_expr
    
    def build(self) -> 'ThermoExprBuilder':
        """
        Build all thermodynamic expressions.
        
        This populates g_ref, g_ideal, g_excess, g_liquid, s_liquid, and h_liquid.
        
        Returns:
            self (for method chaining)
        """
        # Create placeholder reference Gibbs if not set
        if self.g_ref_exprs is None:
            self.g_ref_exprs = tuple(
                sp.Symbol(f'G_ref_{i}') for i in range(self.n_components)
            )
        
        # Create placeholder L expressions if not set
        if self.binary_L_exprs is None:
            self.binary_L_exprs = {}
            for i, j in self.binary_pairs:
                self.binary_L_exprs[(i, j)] = (
                    sp.Symbol(f'L0_{i}{j}'),
                    sp.Symbol(f'L1_{i}{j}')
                )
        
        # Default binary weights = 1
        if self.binary_weights is None:
            self.binary_weights = {pair: sp.Integer(1) for pair in self.binary_pairs}
        
        # Build reference Gibbs
        self.g_ref = self._build_reference_gibbs()
        
        # Build ideal mixing Gibbs
        self.g_ideal = self._build_ideal_mixing_gibbs()
        
        # Build excess Gibbs from binaries
        g_excess_terms = []
        for pair in self.binary_pairs:
            i, j = pair
            l0, l1 = self.binary_L_exprs.get(pair, (sp.Integer(0), sp.Integer(0)))
            weight = self.binary_weights.get(pair, sp.Integer(1))
            
            g_xs_ij = self._build_binary_excess_gibbs(i, j, l0, l1, weight)
            g_excess_terms.append(g_xs_ij)
        
        self.g_excess = sum(g_excess_terms)
        
        # Add higher-order terms
        for indices, expr in self.higher_order_exprs.items():
            product_term = sp.Integer(1)
            for idx in indices:
                product_term *= self.atom_fractions[idx]
            self.g_excess += expr * product_term
        
        # Total Gibbs energy
        self.g_liquid = self.g_ref + self.g_ideal + self.g_excess
        
        # Entropy: S = -dG/dT
        self.s_liquid = -sp.diff(self.g_liquid, self.t)
        
        # Enthalpy: H = G + T*S
        self.h_liquid = self.g_liquid + self.t * self.s_liquid
        
        return self
    
    def get_expressions_dict(self) -> Dict[str, sp.Expr]:
        """
        Get all expressions as a dictionary.
        
        Returns:
            Dictionary with keys: 'g_ref', 'g_ideal', 'g_excess', 'g_liquid', 's_liquid', 'h_liquid'
        """
        return {
            'g_ref': self.g_ref,
            'g_ideal': self.g_ideal,
            'g_excess': self.g_excess,
            'g_liquid': self.g_liquid,
            's_liquid': self.s_liquid,
            'h_liquid': self.h_liquid,
        }
    
    def lambdify(self, expr_name: str = 'g_liquid') -> Callable:
        """
        Create a numpy-callable function from an expression.
        
        Args:
            expr_name: Name of expression to lambdify ('g_liquid', 's_liquid', 'h_liquid', etc.)
        
        Returns:
            Function that takes (x0, x1, ..., x[n-2], T) as arguments.
        """
        expr = getattr(self, expr_name)
        if expr is None:
            raise ValueError(f"Expression '{expr_name}' not built yet. Call build() first.")
        
        args = list(self.x) + [self.t]
        return sp.lambdify(args, expr, 'numpy')


# ==============================================================================
# GENERAL N-COMPONENT INTERPOLATION CLASS
# ==============================================================================

class GeneralInterpolation:
    """
    General n-component liquid solution interpolation.
    
    Takes a chemical system (list of element symbols) and binary interaction
    parameters to compute thermodynamic GTX data over a composition grid.
    Also retrieves DFT formation energies of solid phases from Materials Project.
    
    This class uses ThermoExprBuilder internally for symbolic expression building
    but does not inherit from it - it's a composition-based design.
    
    Attributes:
        elements: List of element symbols in sorted alphabetical order.
        n_components: Number of components in the system.
        output_dir: Directory for caching MP data and outputs.
        binary_pairs: List of binary pair strings (e.g., ["Al-Cu", "Al-Mg"]).
        gtx_data: DataFrame containing the final GTX dataset.
        grid_delta: Composition grid spacing (default 0.025).
    Example:
        >>> interp = GeneralInterpolation(
        ...     elements=['Al', 'Cu', 'Si', 'Mg'],
        ...     output_dir='./output'
        ... )
        >>> interp.set_binary_params({
        ...     'Al-Cu': [1000, 0, 500, 0],
        ...     'Al-Mg': [1500, 0, 700, 0],
        ...     ...
        ... })
        >>> interp.interpolate()
        >>> df = interp.gtx_data
    """
    
    def __init__(
        self,
        elements: List[str],
        output_dir: str,
        grid_delta: float = 0.025,
        temp_delta_k: float = 5.0,
        temp_bounds_offset_k: float = 200.0,
        temp_bounds_k: Optional[Tuple[float, float]] = None,
        param_format: str = 'linear',
        interp_scheme: str = 'linear',
        include_polymorphs: bool = True,
        include_ref_solid_solutions: bool = False,
        ref_solid_solutions_path: Optional[str] = None,
        ref_ss_interp_scheme: str = 'linear',
        tau: float = 8000,
        mp_api_key: Optional[str] = None
    ):
        """
        Initialize the interpolation system.
        
        Args:
            elements: List of element symbols (e.g., ['Al', 'Cu', 'Si', 'Mg']).
            output_dir: Directory for caching data and saving outputs.
            grid_delta: Composition grid spacing (default 0.025).
            temp_delta_k: Temperature grid spacing in Kelvin (default 5 K).
            temp_bounds_offset_k: Default extension from fusion extrema in K (default 200 K).
            temp_bounds_k: Optional explicit temperature bounds (Tmin, Tmax) in Kelvin.
            param_format: L parameter format ('linear', 'exponential', 'combined').
            interp_scheme: Interpolation scheme ('linear', 'muggianu', 'kohler').
            include_polymorphs: If True, include elemental polymorph solid
                references from phase_transitions.json and skip duplicate MP
                pure-element placeholders; if False, keep MP-only references.
            include_ref_solid_solutions: If True, generate and include reference
                solid-solution GTX clouds (BCC/FCC/HCP) from omegas data.
            ref_solid_solutions_path: Optional path to omegas.json.
            ref_ss_interp_scheme: Interpolation scheme for pairwise omega
                interactions in reference solid-solution phases.
            tau: exp constant for combined format (default 8000).
            mp_api_key: Materials Project API key (uses default if None).
        """
        # Sort elements alphabetically for consistency
        self.elements = sorted(elements)
        self.n_components = len(self.elements)
        
        # Interpolation requires at least 3 components (ternary or higher)
        # For binary systems, use BinaryLiquid directly instead
        if self.n_components < 3:
            raise ValueError(
                f"GeneralInterpolation requires at least 3 components (got {self.n_components}). "
                "Binary systems should use BinaryLiquid directly, as there is no interpolation needed."
            )
        
        self.output_dir = output_dir
        self.grid_delta = grid_delta
        self.temp_delta_k = float(temp_delta_k)
        self.temp_bounds_offset_k = float(temp_bounds_offset_k)
        self.temp_bounds_k = temp_bounds_k
        self.param_format = param_format
        self.interp_scheme = interp_scheme
        self.include_polymorphs = bool(include_polymorphs)
        self.include_ref_solid_solutions = bool(include_ref_solid_solutions)
        self.ref_solid_solutions_path = ref_solid_solutions_path
        self.ref_ss_interp_scheme = ref_ss_interp_scheme
        self.tau = tau

        if self.ref_ss_interp_scheme not in INTERPOLATION_SCHEMES:
            raise ValueError(
                f"Unknown ref_ss_interp_scheme '{self.ref_ss_interp_scheme}'. "
                f"Available: {list(INTERPOLATION_SCHEMES.keys())}"
            )
        
        # Generate binary pairs as element strings (sorted alphabetically)
        self._binary_pairs_indices = list(combinations(range(self.n_components), 2))
        self.binary_pairs = [
            f"{self.elements[i]}-{self.elements[j]}" 
            for i, j in self._binary_pairs_indices
        ]
        
        # Create mapping from string pairs to index pairs
        self._pair_str_to_idx = {
            pair: idx_pair 
            for pair, idx_pair in zip(self.binary_pairs, self._binary_pairs_indices)
        }
        
        # Materials Project client
        self._mp_api_key = mp_api_key
        self._mpr = None  # Lazy initialization
        
        # Data storage
        self.fusion_data: Optional[Dict[str, Dict[str, float]]] = None
        self.binary_L_params: Optional[Dict[str, List[float]]] = None
        self.binary_fit_types: Optional[Dict[str, str]] = None
        self.higher_order_params: Dict[Tuple[int, ...], List[float]] = {}
        
        # Output storage
        self.liquid_gtx: Optional[pd.DataFrame] = None
        self.solid_gtx: Optional[pd.DataFrame] = None
        self.gtx_data: Optional[pd.DataFrame] = None
        self.liquid_hsx: Optional[pd.DataFrame] = None
        self.solid_hsx: Optional[pd.DataFrame] = None
        self.hsx_data: Optional[pd.DataFrame] = None
        self.ref_solid_solution_hsx: Optional[pd.DataFrame] = None
        self.ref_solid_solution_gtx: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, any] = {}
        
        # Internal builder (created during interpolation)
        self._expr_builder: Optional[ThermoExprBuilder] = None
        
        print(f"Initialized GeneralInterpolation for system: {'-'.join(self.elements)}")
        print(f"  Components: {self.n_components}")
        print(f"  Binary pairs: {self.binary_pairs}")
        print(f"  Include polymorph references: {self.include_polymorphs}")
        print(f"  Include ref solid solutions: {self.include_ref_solid_solutions}")
    
    @property
    def mpr(self) -> MPRester:
        """Lazy initialization of Materials Project client."""
        if self._mpr is None:
            if self._mp_api_key:
                self._mpr = MPRester(self._mp_api_key)
            else:
                self._mpr = mpr  # Use module-level default
        return self._mpr
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_fusion_data(self) -> 'GeneralInterpolation':
        """
        Load liquid reference enthalpy/entropy/temperature from phase_transitions.json.
        
        Returns:
            self (for method chaining)
        """
        with open(phase_transitions_file) as f:
            transitions_raw = json.load(f)

        elem_map = transitions_raw.get('elements', transitions_raw)

        liquid_enthalpies: Dict[str, float] = {}
        liquid_entropies: Dict[str, float] = {}
        melt_temps: Dict[str, float] = {}

        for symbol, elem_data in elem_map.items():
            for phase in elem_data.get('phases', []):
                if phase.get('phase_type') != 'liquid':
                    continue
                h_val = phase.get('enthalpy_J_per_mol')
                s_val = phase.get('entropy_J_per_mol_K')
                t_val = phase.get('transition_temperature_K')
                if h_val is not None:
                    liquid_enthalpies[symbol] = float(h_val)
                if s_val is not None:
                    liquid_entropies[symbol] = float(s_val)
                if t_val is not None:
                    melt_temps[symbol] = float(t_val)

        missing_h = [el for el in self.elements if el not in liquid_enthalpies]
        missing_s = [el for el in self.elements if el not in liquid_entropies]
        missing_t = [el for el in self.elements if el not in melt_temps]
        if missing_h or missing_s or missing_t:
            missing_parts = []
            if missing_h:
                missing_parts.append(f"enthalpy: {missing_h}")
            if missing_s:
                missing_parts.append(f"entropy: {missing_s}")
            if missing_t:
                missing_parts.append(f"Tm: {missing_t}")
            raise ValueError("Missing liquid reference data in phase_transitions.json -> " + "; ".join(missing_parts))

        self.fusion_data = {
            'enthalpies': [liquid_enthalpies[el] for el in self.elements],
            'entropies': [liquid_entropies[el] for el in self.elements],
            'temperatures': [melt_temps[el] for el in self.elements],
        }
        
        return self

    def _load_phase_transition_elements(self) -> Dict[str, dict]:
        """Load phase transition element map from config file."""
        with open(phase_transitions_file) as f:
            transitions_raw = json.load(f)
        return transitions_raw.get('elements', transitions_raw)

    def _build_elemental_polymorph_df(self) -> pd.DataFrame:
        """Build elemental solid polymorph reference rows for the current system.

        Each polymorph is represented as a solid phase at pure-element composition
        with its own H and S from phase_transitions.json.
        """
        elements_map = self._load_phase_transition_elements()
        rows: list[dict] = []

        for el_idx, el in enumerate(self.elements):
            elem_data = elements_map.get(el, {})
            for phase in elem_data.get('phases', []):
                if phase.get('phase_type') != 'solid':
                    continue

                # Match existing ternary reference behavior: exclude ground-state marker entries.
                t_trans = phase.get('transition_temperature_K')
                if t_trans is None or float(t_trans) < 0:
                    continue

                h_val = phase.get('enthalpy_J_per_mol')
                s_val = phase.get('entropy_J_per_mol_K')
                phase_name = phase.get('common_name') or phase.get('phase_name')
                if h_val is None or s_val is None or not phase_name:
                    continue

                comp = np.zeros(self.n_components, dtype=float)
                comp[el_idx] = 1.0

                row = {f'x{i}': float(comp[i + 1]) for i in range(self.n_components - 1)}
                row['H'] = float(h_val)
                row['S'] = float(s_val)
                row['Phase Name'] = str(phase_name)
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.get_composition_columns() + ['H', 'S', 'Phase Name'])

        poly_df = pd.DataFrame(rows)
        poly_df = poly_df.drop_duplicates(subset=self.get_composition_columns() + ['Phase Name', 'H', 'S'])
        return poly_df.reset_index(drop=True)

    def _element_indices_with_polymorph_refs(self) -> set[int]:
        """Return canonical element indices with at least one valid solid polymorph ref."""
        elements_map = self._load_phase_transition_elements()
        indices: set[int] = set()

        for el_idx, el in enumerate(self.elements):
            elem_data = elements_map.get(el, {})
            for phase in elem_data.get('phases', []):
                if phase.get('phase_type') != 'solid':
                    continue
                t_trans = phase.get('transition_temperature_K')
                if t_trans is None or float(t_trans) < 0:
                    continue
                h_val = phase.get('enthalpy_J_per_mol')
                s_val = phase.get('entropy_J_per_mol_K')
                phase_name = phase.get('common_name') or phase.get('phase_name')
                if h_val is None or s_val is None or not phase_name:
                    continue
                indices.add(el_idx)
                break

        return indices

    def _resolve_ref_ss_omegas_path(self) -> Path:
        """Resolve the omegas data path for reference solid solutions."""
        if self.ref_solid_solutions_path:
            return Path(self.ref_solid_solutions_path)
        return DEFAULT_REF_SS_OMEGAS_PATH

    def _load_ref_ss_omegas_data(self) -> dict:
        """Load and validate omegas.json data required for ref solid solutions."""
        omegas_path = self._resolve_ref_ss_omegas_path()
        if not omegas_path.exists():
            raise FileNotFoundError(f"Reference solid-solution omegas file not found: {omegas_path}")

        with open(omegas_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'omegas' not in data or 'elements' not in data:
            raise ValueError("omegas.json must contain top-level keys 'omegas' and 'elements'.")

        return data

    def _find_template_transition_temp_k(self, el: str, template: str, pt_map: Dict[str, dict]) -> Optional[float]:
        """Find transition temperature for an element's template phase from phase transitions DB."""
        elem_data = pt_map.get(el, {})
        for phase in elem_data.get('phases', []):
            if phase.get('phase_type') != 'solid':
                continue
            t_val = phase.get('transition_temperature_K')
            if t_val is None or float(t_val) <= 0.0:
                continue

            sg_num = phase.get('spacegroup_number')
            phase_name = str(phase.get('common_name') or phase.get('phase_name') or '').lower()
            if sg_num == REF_SS_SPACEGROUPS[template] or any(
                marker in phase_name for marker in REF_SS_PHASE_NAME_MARKERS[template]
            ):
                return float(t_val)

        return None

    def _compute_ref_ss_endpoint_offsets(
        self,
        omega_data: dict,
        pt_map: Dict[str, dict],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute endpoint DeltaH/DeltaS vectors for each template phase.

        DeltaH values are referenced to the minimum elemental energy among templates.
        DeltaS values use a cumulative stepwise DeltaH/T accumulation consistent with
        fitting_demo_solid_solution.py legacy resolver behavior.
        """
        element_blocks = omega_data.get('elements', {})
        per_element_template_energy: Dict[str, Dict[str, float]] = {el: {} for el in self.elements}

        for template in REF_SS_PHASES:
            template_block = element_blocks.get(template, {})
            if not isinstance(template_block, dict):
                continue
            for el in self.elements:
                if el in template_block:
                    per_element_template_energy[el][template] = float(template_block[el])

        ground_energy: Dict[str, float] = {}
        for el in self.elements:
            energies = list(per_element_template_energy[el].values())
            if not energies:
                continue
            ground_energy[el] = min(energies)

        endpoint_offsets: Dict[str, Dict[str, np.ndarray]] = {}
        for template in REF_SS_PHASES:
            if template not in element_blocks:
                continue

            if any(template not in per_element_template_energy[el] for el in self.elements):
                continue

            delta_h = np.zeros(self.n_components, dtype=float)
            delta_s = np.zeros(self.n_components, dtype=float)

            for i, el in enumerate(self.elements):
                ss_energy = per_element_template_energy[el][template]
                d_h = (ss_energy - ground_energy[el]) * EV_PER_ATOM_TO_J_PER_MOL
                delta_h[i] = float(d_h)

                t_this = self._find_template_transition_temp_k(el, template, pt_map)
                if t_this is None or t_this <= 0.0:
                    delta_s[i] = 0.0
                    continue

                # Cumulative stepwise entropy contribution up to current template.
                steps = []
                for phase_k, e_val in per_element_template_energy[el].items():
                    t_k = self._find_template_transition_temp_k(el, phase_k, pt_map)
                    if t_k is None or t_k <= 0.0 or t_k > t_this + 1e-12:
                        continue
                    d_h_k = (e_val - ground_energy[el]) * EV_PER_ATOM_TO_J_PER_MOL
                    steps.append((float(t_k), float(d_h_k)))

                if not steps:
                    delta_s[i] = 0.0
                    continue

                steps.sort(key=lambda x: x[0])
                s_accum = 0.0
                prev_h = 0.0
                for t_k, d_h_k in steps:
                    s_accum += (d_h_k - prev_h) / t_k
                    prev_h = d_h_k
                delta_s[i] = float(s_accum)

            endpoint_offsets[template] = {
                'delta_h_vec': delta_h,
                'delta_s_vec': delta_s,
            }

        return endpoint_offsets

    def _ref_ss_pair_weight(self, comps: np.ndarray, i: int, j: int) -> np.ndarray:
        """Pair interaction weighting for ref solid solutions.

        For pairwise omega-only regular-solution terms, interpolation schemes map
        to the same xi*xj prefactor; keep this method for explicit scheme gating.
        """
        _ = self.ref_ss_interp_scheme
        return comps[:, i] * comps[:, j]

    def compute_ref_solid_solution_hsx(self) -> pd.DataFrame:
        """Generate reference BCC/FCC/HCP solid-solution HSX clouds from omegas data."""
        if not self.include_ref_solid_solutions:
            self.ref_solid_solution_hsx = pd.DataFrame(
                columns=self.get_composition_columns() + ['H', 'S', 'Phase Name']
            )
            return self.ref_solid_solution_hsx

        omega_raw = self._load_ref_ss_omegas_data()
        omega_blocks = omega_raw.get('omegas', {})
        pt_map = self._load_phase_transition_elements()
        endpoint_offsets = self._compute_ref_ss_endpoint_offsets(omega_raw, pt_map)

        comps = self._generate_composition_grid()
        system_label = '-'.join(self.elements)

        phase_frames = []
        included_templates = []
        for template in REF_SS_PHASES:
            if template not in endpoint_offsets:
                continue
            omega_phase = omega_blocks.get(template, {})
            if not isinstance(omega_phase, dict):
                continue

            omega_pair_map: Dict[Tuple[int, int], float] = {}
            missing_pairs = []
            for i, j in self._binary_pairs_indices:
                pair_key = f"{self.elements[i]}-{self.elements[j]}"
                if pair_key not in omega_phase:
                    missing_pairs.append(pair_key)
                    continue
                omega_pair_map[(i, j)] = float(omega_phase[pair_key]) * EV_PER_ATOM_TO_J_PER_MOL

            if missing_pairs:
                self.metadata[f'ref_ss_missing_pairs_{template}'] = missing_pairs
                continue

            included_templates.append(template)
            delta_h_vec = endpoint_offsets[template]['delta_h_vec']
            delta_s_vec = endpoint_offsets[template]['delta_s_vec']

            h_offsets = comps @ delta_h_vec
            s_offsets = comps @ delta_s_vec

            with np.errstate(divide='ignore', invalid='ignore'):
                log_terms = np.where(comps > 0.0, comps * np.log(comps), 0.0)
            s_conf = -R * np.sum(log_terms, axis=1)

            h_mix = np.zeros(len(comps), dtype=float)
            for (i, j), omega_val in omega_pair_map.items():
                h_mix += omega_val * self._ref_ss_pair_weight(comps, i, j)

            h_total = h_offsets + h_mix
            s_total = s_offsets + s_conf

            frame = pd.DataFrame({
                **{f'x{k}': comps[:, k + 1] for k in range(self.n_components - 1)},
                'H': h_total,
                'S': s_total,
                'Phase Name': f"{template}__{system_label}",
            })
            phase_frames.append(frame)

        if not phase_frames:
            raise ValueError(
                f"No reference solid-solution templates available for system {system_label}. "
                "Check omegas.json phase blocks and pair coverage."
            )

        self.ref_solid_solution_hsx = pd.concat(phase_frames, ignore_index=True)
        self.ref_solid_solution_hsx = self.ref_solid_solution_hsx.drop_duplicates()
        self.ref_solid_solution_hsx = self.ref_solid_solution_hsx.reset_index(drop=True)

        self.metadata['n_ref_ss_templates_included'] = int(len(included_templates))
        self.metadata['ref_ss_templates_included'] = included_templates
        self.metadata['n_ref_ss_hsx_points'] = int(len(self.ref_solid_solution_hsx))
        return self.ref_solid_solution_hsx

    def _remove_terminal_solid_rows_for_ref_ss(self):
        """Remove pure-element solid rows from solid_hsx when ref SS clouds are enabled."""
        if self.solid_hsx is None or self.solid_hsx.empty:
            self.metadata['n_terminal_solid_rows_removed_for_ref_ss'] = 0
            return

        comp_cols = self.get_composition_columns()
        indep = self.solid_hsx[comp_cols].to_numpy(dtype=float)
        dep = 1.0 - indep.sum(axis=1)
        full = np.column_stack([dep, indep])

        terminal_mask = (full > 1e-10).sum(axis=1) == 1
        removed = int(np.sum(terminal_mask))
        if removed > 0:
            self.solid_hsx = self.solid_hsx.loc[~terminal_mask].reset_index(drop=True)
        self.metadata['n_terminal_solid_rows_removed_for_ref_ss'] = removed

    def compute_ref_solid_solution_gtx(self, temp_grid_k: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Expand reference solid-solution HSX points into GTX over temperature grid."""
        if self.ref_solid_solution_hsx is None:
            self.compute_ref_solid_solution_hsx()

        if temp_grid_k is None:
            temp_grid_k = self._generate_temperature_grid()

        if self.ref_solid_solution_hsx is None or self.ref_solid_solution_hsx.empty:
            self.ref_solid_solution_gtx = pd.DataFrame(
                columns=self.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase Name']
            )
            return self.ref_solid_solution_gtx

        frames = []
        for temp_k in temp_grid_k:
            df_t = self.ref_solid_solution_hsx.copy()
            df_t['T_K'] = float(temp_k)
            df_t['T_C'] = float(temp_k - 273.15)
            df_t['G'] = df_t['H'].astype(float) - float(temp_k) * df_t['S'].astype(float)
            keep_cols = self.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase Name']
            frames.append(df_t[keep_cols])

        self.ref_solid_solution_gtx = pd.concat(frames, ignore_index=True)
        self.metadata['n_ref_ss_gtx_points'] = int(len(self.ref_solid_solution_gtx))
        return self.ref_solid_solution_gtx
    
    def set_binary_params(
        self,
        L_params: Dict[str, List[float]],
        fit_types: Optional[Dict[str, str]] = None
    ) -> 'GeneralInterpolation':
        """
        Set binary interaction parameters.
        
        Args:
            L_params: Dict mapping binary pair strings to [L0_a, L0_b, L1_a, L1_b].
                Keys should be element pairs like "Al-Cu" (alphabetically sorted).
            fit_types: Optional dict mapping pair strings to "fit" or "pred".
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If not all binary pairs have parameters.
        """
        normalized_params: Dict[str, List[float]] = {}
        for key, raw_params in L_params.items():
            parts = [p.strip() for p in key.split('-')]
            if len(parts) != 2:
                raise ValueError(f"Invalid binary key '{key}'. Expected format 'A-B'.")
            if len(raw_params) != 4:
                raise ValueError(f"Expected 4 parameters for '{key}' [L0_a, L0_b, L1_a, L1_b], got {len(raw_params)}")

            canonical_key = '-'.join(sorted(parts))
            params = [float(v) for v in raw_params]

            # If the provided key order is reversed relative to canonical ordering,
            # map to canonical key and flip L1 sign (same convention as exec_tern_single.py).
            if key != canonical_key:
                params[2] = -params[2]
                params[3] = -params[3]

            if canonical_key in normalized_params:
                existing = normalized_params[canonical_key]
                if not np.allclose(existing, params, rtol=1e-12, atol=1e-12):
                    raise ValueError(
                        f"Conflicting parameters for binary pair '{canonical_key}'. "
                        f"Received both {existing} and {params}."
                    )
            normalized_params[canonical_key] = params
        
        # Check all pairs are provided
        missing = set(self.binary_pairs) - set(normalized_params.keys())
        if missing:
            raise ValueError(f"Missing L parameters for binary pairs: {missing}")
        
        self.binary_L_params = normalized_params
        self.binary_fit_types = fit_types or {pair: 'pred' for pair in self.binary_pairs}

        # Normalize fit-type keys if provided.
        if fit_types:
            normalized_fit_types: Dict[str, str] = {}
            for key, label in fit_types.items():
                parts = [p.strip() for p in key.split('-')]
                if len(parts) != 2:
                    continue
                canonical_key = '-'.join(sorted(parts))
                normalized_fit_types[canonical_key] = label
            for pair in self.binary_pairs:
                normalized_fit_types.setdefault(pair, 'pred')
            self.binary_fit_types = normalized_fit_types
        
        return self
    
    def add_higher_order_params(
        self,
        component_indices: Tuple[int, ...],
        params: List[float]
    ) -> 'GeneralInterpolation':
        """
        Add higher-order interaction parameters (ternary, quaternary, etc.).
        
        Args:
            component_indices: Tuple of component indices (e.g., (0, 1, 2) for ternary).
            params: Parameter vector. Default supported format is
                [L2_a, L2_b, L3_a, L3_b], which is internally transformed
                to temperature-dependent L2(T), L3(T) expressions.
        
        Returns:
            self (for method chaining)
        """
        self.higher_order_params[tuple(sorted(component_indices))] = params
        return self
    
    # =========================================================================
    # Composition Grid Generation
    # =========================================================================
    
    def _generate_composition_grid(self) -> np.ndarray:
        """
        Generate composition grid using independent variables for latter components.

        For n components in canonical order [E0, E1, ..., E{n-1}]:
        - Independent variables are x(E1), x(E2), ..., x(E{n-1}).
        - Dependent variable is x(E0) = 1 - sum(independent).
        
        Returns:
            Array of shape (n_points, n_components) in canonical component order.
        """
        steps = np.arange(0, 1 + self.grid_delta, self.grid_delta)

        # Independent grid: dimensions n_components - 1.
        indep_grids = np.meshgrid(*[steps] * (self.n_components - 1), indexing='ij')
        indep_points = np.stack([g.flatten() for g in indep_grids], axis=1)

        indep_sums = indep_points.sum(axis=1)
        valid_mask = (indep_sums <= 1.0 + 1e-9)
        indep_valid = indep_points[valid_mask]

        # Keep higher precision for non-decimal grid steps (e.g. 0.025), then
        # recompute dependent composition from rounded independent variables.
        indep_valid = np.round(indep_valid, 12)
        dep = 1.0 - indep_valid.sum(axis=1)

        # Drop only clearly non-physical points from floating point noise.
        keep = dep >= -1e-12
        indep_valid = indep_valid[keep]
        dep = dep[keep]
        dep = np.clip(dep, 0.0, 1.0)

        valid_compositions = np.column_stack([dep, indep_valid])
        valid_compositions = np.round(valid_compositions, 12)

        return valid_compositions

    def _generate_temperature_grid(self) -> np.ndarray:
        """Generate a temperature grid in Kelvin for GTX evaluation."""
        if self.fusion_data is None:
            self.load_fusion_data()

        if self.temp_bounds_k is not None:
            t_min_k, t_max_k = self.temp_bounds_k
        else:
            t_melts = np.asarray(self.fusion_data['temperatures'], dtype=float)
            t_min_k = float(np.min(t_melts) - self.temp_bounds_offset_k)
            t_max_k = float(np.max(t_melts) + self.temp_bounds_offset_k)

        t_min_k = max(0.0, float(t_min_k))
        t_max_k = max(t_min_k, float(t_max_k))

        if self.temp_delta_k <= 0:
            raise ValueError("temp_delta_k must be positive.")

        t_grid = np.arange(t_min_k, t_max_k + 0.5 * self.temp_delta_k, self.temp_delta_k)
        return np.round(t_grid, 8)
    
    # =========================================================================
    # Liquid  Computation
    # =========================================================================
    
    def _build_expressions(self) -> ThermoExprBuilder:
        """Build thermodynamic expressions using ThermoExprBuilder."""
        if self.fusion_data is None:
            self.load_fusion_data()
        
        builder = ThermoExprBuilder(
            n_components=self.n_components,
            param_format=self.param_format,
            interp_scheme=self.interp_scheme,
            tau=self.tau
        )
        
        # Set reference Gibbs energies directly from liquid H/S values.
        g_ref_exprs = tuple(
            self.fusion_data['enthalpies'][i] - builder.t * self.fusion_data['entropies'][i]
            for i in range(self.n_components)
        )
        builder.set_reference_gibbs(g_ref_exprs)
        
        # Convert string-keyed params to index-keyed params
        if self.binary_L_params is not None:
            index_params = {
                self._pair_str_to_idx[pair]: params
                for pair, params in self.binary_L_params.items()
            }
            builder.set_binary_L_params(index_params)
        
        # Add higher-order terms
        for indices, params in self.higher_order_params.items():
            if len(params) == 4:
                # Default higher-order vector format:
                # [L2_a, L2_b, L3_a, L3_b] with the same param_format as binaries.
                expr_func = PARAM_EXPR_REGISTRY[self.param_format]
                if self.param_format == 'combined':
                    l2_expr = _combined_expr(params[0], params[1], builder.t, builder.tau)
                    l3_expr = _combined_expr(params[2], params[3], builder.t, builder.tau)
                else:
                    l2_expr = expr_func(params[0], params[1], builder.t)
                    l3_expr = expr_func(params[2], params[3], builder.t)

                x_first = builder.get_atom_fraction(indices[0])
                x_last = builder.get_atom_fraction(indices[-1])
                expr = l2_expr + l3_expr * (x_first - x_last)
            elif len(params) == 2:
                # Backward-compatible fallback: [const, linear_T_coeff].
                expr = params[0] + params[1] * builder.t
            elif len(params) == 1:
                expr = params[0]
            else:
                raise ValueError(
                    f"Unsupported higher-order parameter vector for indices {indices}: {params}. "
                    "Expected lengths 1, 2, or 4."
                )

            builder.add_higher_order_term(indices, expr)
        
        builder.build()
        return builder
    
    def compute_liquid_gtx(self) -> pd.DataFrame:
        """
        Compute liquid phase GTX data on composition and temperature grids.
        
        Returns:
            DataFrame with columns: x0, x1, ..., x[n-2], T_K, T_C, G, 'Phase Name'
        """
        self._expr_builder = self._build_expressions()

        compositions = self._generate_composition_grid()
        t_grid_k = self._generate_temperature_grid()

        g_func = self._expr_builder.lambdify('g_liquid')
        x_indep = [compositions[:, i + 1] for i in range(self.n_components - 1)]

        gtx_frames = []
        for temp_k in t_grid_k:
            eval_args = x_indep + [temp_k]
            with np.errstate(divide='ignore', invalid='ignore'):
                g_values = g_func(*eval_args)

            g_values = np.asarray(g_values, dtype=float)
            g_values = np.where(np.isfinite(g_values), g_values, np.nan)

            data = {}
            for i in range(self.n_components - 1):
                # x0.. are the latter canonical components (E1..E{n-1}).
                data[f'x{i}'] = compositions[:, i + 1]
            data['T_K'] = float(temp_k)
            data['T_C'] = float(temp_k - 273.15)
            data['G'] = g_values
            data['Phase Name'] = 'L'
            gtx_frames.append(pd.DataFrame(data))

        self.liquid_gtx = pd.concat(gtx_frames, ignore_index=True)
        self.liquid_gtx = self.liquid_gtx.dropna(subset=['G']).reset_index(drop=True)
        return self.liquid_gtx

    def compute_liquid_hsx(self) -> pd.DataFrame:
        """Deprecated HSX helper retained for backward compatibility."""
        raise NotImplementedError(
            "compute_liquid_hsx has been replaced by compute_liquid_gtx (G(T,x) workflow)."
        )
    
    # =========================================================================
    # Solid Formation Energies from Materials Project
    # =========================================================================
    
    def _get_mp_entries(self, use_cache: bool = True) -> List[dict]:
        """
        Fetch computed structure entries from Materials Project.
        
        Args:
            use_cache: If True, load from cache file if available.
        
        Returns:
            List of serialized ComputedStructureEntry dicts.
        """
        # Ensure output directory exists
        cache_dir = os.path.join(self.output_dir, 'mp_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        system_name = '-'.join(self.elements)
        cache_path = os.path.join(cache_dir, f"{system_name}_entries.json")
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading MP entries from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        print(f"Fetching MP entries for {system_name}...")
        entries = self.mpr.get_entries_in_chemsys(self.elements)
        sanitized = jsanitize(entries)
        
        with open(cache_path, 'w') as f:
            json.dump(sanitized, f)
        
        return sanitized
    
    def fetch_solid_formation_energies(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch formation energies of stable solid phases from Materials Project.
        
        Args:
            use_cache: If True, use cached MP data if available.
        
        Returns:
            DataFrame with columns: x0, x1, ..., x[n-2], H, S, 'Phase Name'
        """
        entries_data = self._get_mp_entries(use_cache)
        entries = [ComputedStructureEntry.from_dict(e) for e in entries_data]
        
        # Filter problematic entries (e.g., Mg with 149 atoms)
        if "Mg" in self.elements:
            entries = [e for e in entries if e.composition.get("Mg", 0) != 149]
        
        # Build phase diagram
        phase_diagram = PhaseDiagram(entries)
        stable_entries = phase_diagram.stable_entries
        
        # Count compounds with all components present
        n_full_compounds = sum(
            1 for e in stable_entries 
            if len(e.composition.elements) == self.n_components
        )
        self.metadata['n_full_compounds'] = n_full_compounds
        
        # Extract formation energies and compositions
        phase_names = []
        compositions_list = []
        formation_energies = []
        skipped_mp_elementals = 0
        
        element_objects = [Element(el) for el in self.elements]
        
        polymorph_element_indices = self._element_indices_with_polymorph_refs() if self.include_polymorphs else set()

        for entry in stable_entries:
            formula = entry.composition.reduced_formula
            comp = Composition(formula)
            entry_elements = comp.elements
            
            # Formation energy in J/mol (converted from eV/atom).
            form_energy = phase_diagram.get_form_energy_per_atom(entry) * EV_PER_ATOM_TO_J_PER_MOL
            
            # Get atomic fractions for each element
            atom_fracs = []
            for element in element_objects:
                if element in entry_elements:
                    atom_fracs.append(comp.get_atomic_fraction(element))
                else:
                    atom_fracs.append(0.0)

            # Mirror binary.py endpoint handling: if explicit elemental
            # polymorph references exist, skip MP pure-element placeholders.
            nz_idx = [i for i, v in enumerate(atom_fracs) if v > 1e-10]
            if len(nz_idx) == 1 and nz_idx[0] in polymorph_element_indices:
                skipped_mp_elementals += 1
                continue
            
            compositions_list.append(atom_fracs)
            phase_names.append(formula)
            formation_energies.append(form_energy)
        
        if compositions_list:
            compositions_arr = np.array(compositions_list, dtype=float)
        else:
            compositions_arr = np.empty((0, self.n_components), dtype=float)
        
        # Track deepest formation energy
        self.metadata['deepest_formation_energy'] = min(formation_energies) if formation_energies else None
        
        # Build DataFrame using independent composition convention:
        # x0.. map to canonical components E1..E{n-1}.
        data = {}
        for i in range(self.n_components - 1):
            data[f'x{i}'] = compositions_arr[:, i + 1]
        data['H'] = formation_energies
        data['S'] = [0.0] * len(formation_energies)  # Solids have S=0 in this model
        data['Phase Name'] = phase_names
        
        df = pd.DataFrame(data)
        
        # Keep only lowest energy entry for each phase
        df = df.loc[df.groupby('Phase Name')['H'].idxmin()]
        df = df.reset_index(drop=True)
        
        # Add elemental polymorph references (same composition, different G(T)).
        if self.include_polymorphs:
            poly_df = self._build_elemental_polymorph_df()
        else:
            poly_df = pd.DataFrame(columns=self.get_composition_columns() + ['H', 'S', 'Phase Name'])
        if not poly_df.empty:
            df = pd.concat([df, poly_df], ignore_index=True)
            df = df.drop_duplicates(subset=self.get_composition_columns() + ['Phase Name', 'H', 'S'])
            df = df.reset_index(drop=True)

        self.solid_hsx = df
        self.metadata['include_polymorphs'] = bool(self.include_polymorphs)
        self.metadata['n_elemental_polymorph_refs'] = int(len(poly_df))
        self.metadata['n_mp_elemental_entries_skipped_for_polymorphs'] = int(skipped_mp_elementals)
        return self.solid_hsx

    def compute_solid_gtx(self, temp_grid_k: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Expand solid reference energies to GTX by evaluating G = H - T*S over T grid."""
        if self.solid_hsx is None:
            self.fetch_solid_formation_energies(use_cache=True)

        if temp_grid_k is None:
            temp_grid_k = self._generate_temperature_grid()

        frames = []
        for temp_k in temp_grid_k:
            df_t = self.solid_hsx.copy()
            df_t['T_K'] = float(temp_k)
            df_t['T_C'] = float(temp_k - 273.15)
            df_t['G'] = df_t['H'].astype(float) - float(temp_k) * df_t['S'].astype(float)
            keep_cols = self.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase Name']
            frames.append(df_t[keep_cols])

        self.solid_gtx = pd.concat(frames, ignore_index=True)
        return self.solid_gtx
    
    # =========================================================================
    # Main Interpolation Pipeline
    # =========================================================================
    
    def interpolate(self, use_mp_cache: bool = True) -> 'GeneralInterpolation':
        """
        Run the full interpolation pipeline.
        
        This method:
        1. Loads fusion data (if not already loaded)
        2. Computes liquid phase GTX on composition/temperature grids
        3. Fetches solid formation energies from Materials Project
        4. Optionally generates reference solid-solution clouds (BCC/FCC/HCP)
        5. Combines all enabled phase datasets into final GTX
        
        Args:
            use_mp_cache: Whether to use cached MP data.
        
        Returns:
            self (for method chaining)
        """
        print(f"\n{'='*60}")
        print(f"Running interpolation for {'-'.join(self.elements)}")
        print(f"{'='*60}")
        
        # Step 1: Load fusion data
        if self.fusion_data is None:
            print("\n[1/3] Loading fusion data...")
            self.load_fusion_data()
        else:
            print("\n[1/3] Fusion data already loaded")
        
        # Step 2: Compute liquid GTX
        print("\n[2/4] Computing liquid phase GTX...")
        self.compute_liquid_gtx()
        print(f"       Generated {len(self.liquid_gtx)} liquid GTX points")

        # Step 3: Fetch solid formation energies
        print("\n[3/4] Fetching solid formation energies...")
        self.fetch_solid_formation_energies(use_cache=use_mp_cache)
        print(f"       Found {len(self.solid_hsx)} stable solid phases")

        if self.include_ref_solid_solutions:
            print("       Generating reference solid-solution HSX clouds...")
            self.compute_ref_solid_solution_hsx()
            print(f"       Generated {len(self.ref_solid_solution_hsx)} reference HSX points")

            self._remove_terminal_solid_rows_for_ref_ss()
            removed = int(self.metadata.get('n_terminal_solid_rows_removed_for_ref_ss', 0))
            print(f"       Removed {removed} pure-terminal MP solid rows (ref SS mode)")
        else:
            self.ref_solid_solution_hsx = pd.DataFrame(
                columns=self.get_composition_columns() + ['H', 'S', 'Phase Name']
            )
            self.ref_solid_solution_gtx = pd.DataFrame(
                columns=self.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase Name']
            )
            self.metadata['n_terminal_solid_rows_removed_for_ref_ss'] = 0

        # Step 4: Expand solids to GTX and combine.
        print("\n[4/4] Expanding solids to GTX and combining datasets...")
        temp_grid_k = self._generate_temperature_grid()
        self.compute_solid_gtx(temp_grid_k=temp_grid_k)

        frames = [self.liquid_gtx, self.solid_gtx]
        if self.include_ref_solid_solutions:
            self.compute_ref_solid_solution_gtx(temp_grid_k=temp_grid_k)
            print(f"       Generated {len(self.ref_solid_solution_gtx)} reference GTX points")
            frames.append(self.ref_solid_solution_gtx)

        self.gtx_data = pd.concat(frames, ignore_index=True)
        self.gtx_data = self.gtx_data.drop_duplicates()
        self.gtx_data = self.gtx_data.reset_index(drop=True)

        self.metadata['include_ref_solid_solutions'] = bool(self.include_ref_solid_solutions)
        self.metadata['ref_ss_interp_scheme'] = self.ref_ss_interp_scheme
        self.metadata['n_ref_ss_phases'] = int(
            self.ref_solid_solution_gtx['Phase Name'].nunique()
        ) if self.ref_solid_solution_gtx is not None and not self.ref_solid_solution_gtx.empty else 0

        
        print(f"\nInterpolation complete!")
        print(f"  Total GTX points: {len(self.gtx_data)}")
        print(f"  Phases: {self.gtx_data['Phase Name'].nunique()}")
        
        return self
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_composition_columns(self) -> List[str]:
        """Get list of composition column names."""
        return [f'x{i}' for i in range(self.n_components - 1)]
    
    def element_to_index(self, element: str) -> int:
        """Get index of an element in the sorted elements list."""
        return self.elements.index(element)
    
    def index_to_element(self, index: int) -> str:
        """Get element symbol from its index."""
        return self.elements[index]
    
    def get_pair_elements(self, pair_str: str) -> Tuple[str, str]:
        """Parse a binary pair string into element tuple."""
        parts = pair_str.split('-')
        return tuple(sorted(parts))
    
    def save_gtx(self, filename: Optional[str] = None) -> str:
        """
        Save the GTX data to a CSV file.
        
        Args:
            filename: Output filename. If None, uses system name.
        
        Returns:
            Path to saved file.
        """
        if self.gtx_data is None:
            raise ValueError("No GTX data to save. Run interpolate() first.")
        
        if filename is None:
            filename = f"{'-'.join(self.elements)}_gtx.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        self.gtx_data.to_csv(filepath, index=False)
        print(f"Saved GTX data to: {filepath}")
        return filepath
    
    def summary(self) -> Dict[str, any]:
        """
        Get a summary of the current state.
        
        Returns:
            Dictionary with system info and statistics.
        """
        return {
            'system': '-'.join(self.elements),
            'n_components': self.n_components,
            'binary_pairs': self.binary_pairs,
            'grid_delta': self.grid_delta,
            'param_format': self.param_format,
            'interp_scheme': self.interp_scheme,
            'include_polymorphs': self.include_polymorphs,
            'include_ref_solid_solutions': self.include_ref_solid_solutions,
            'ref_ss_interp_scheme': self.ref_ss_interp_scheme,
            'fusion_data_loaded': self.fusion_data is not None,
            'binary_params_set': self.binary_L_params is not None,
            'n_liquid_points': len(self.liquid_gtx) if self.liquid_gtx is not None else 0,
            'n_solid_phases': len(self.solid_hsx) if self.solid_hsx is not None else 0,
            'n_ref_ss_points': len(self.ref_solid_solution_gtx) if self.ref_solid_solution_gtx is not None else 0,
            'temp_delta_k': self.temp_delta_k,
            **self.metadata
        }


class GeneralEquilibrium:
    """General lower-hull equilibrium solver over GTX data.

    This class consumes GTX rows (x*, T, G, phase label), solves lower hull per
    temperature slice, emits simplex-linked equilibrium rows, and provides a
    cluster-aware lowest-liquidus invariant utility.
    """

    def __init__(
        self,
        gtx_data: pd.DataFrame,
        composition_cols: Optional[List[str]] = None,
        temp_col: str = 'T_K',
        phase_col: str = 'Phase Name',
        g_col: str = 'G',
        liquid_aliases: Optional[List[str]] = None,
        cluster_comp_threshold: Optional[float] = None,
        temp_step_k: Optional[float] = None,
    ):
        self.gtx_data = gtx_data.copy()
        self.temp_col = temp_col
        self.phase_col = phase_col
        self.g_col = g_col
        self.composition_cols = composition_cols or self._infer_composition_columns(self.gtx_data)
        self.liquid_aliases = {s.strip().upper() for s in (liquid_aliases or ['L', 'LIQUID'])}

        self.cluster_comp_threshold = cluster_comp_threshold
        self.temp_step_k = temp_step_k

        self.equilibrium_df: Optional[pd.DataFrame] = None
        self.simplex_df: Optional[pd.DataFrame] = None

        # Temperature-sweep progress tracking.
        self.total_temperature_slices: int = 0
        self.processed_temperature_slices: List[float] = []
        self.successful_temperature_slices: List[float] = []
        self.failed_temperature_slices: List[float] = []
        self.temperature_slice_status: Dict[float, str] = {}

        self._validate_input()

    @staticmethod
    def _infer_composition_columns(df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if c.startswith('x') and c[1:].isdigit()]
        if not cols:
            raise ValueError("Could not infer composition columns. Expected columns like x0, x1, ...")
        return sorted(cols, key=lambda c: int(c[1:]))

    def _validate_input(self):
        required = set(self.composition_cols + [self.temp_col, self.phase_col, self.g_col])
        missing = required - set(self.gtx_data.columns)
        if missing:
            raise ValueError(f"Missing required GTX columns for equilibrium solve: {sorted(missing)}")

        if self.gtx_data.empty:
            raise ValueError("GTX dataframe is empty.")

        if not np.isfinite(self.gtx_data[self.g_col].to_numpy(dtype=float)).all():
            raise ValueError("GTX dataframe contains non-finite G values.")

        if not np.isfinite(self.gtx_data[self.temp_col].to_numpy(dtype=float)).all():
            raise ValueError("GTX dataframe contains non-finite temperature values.")

        n_vertices_required = len(self.composition_cols) + 1
        temp_sizes = self.gtx_data.groupby(self.temp_col, sort=True).size()
        if (temp_sizes < n_vertices_required).any():
            bad_t = temp_sizes[temp_sizes < n_vertices_required].index.tolist()[:5]
            raise ValueError(
                f"Temperature slices must contain at least {n_vertices_required} points; "
                f"insufficient slices include: {bad_t}"
            )

    def _is_liquid_label(self, phase_name: Any) -> bool:
        return str(phase_name).strip().upper() in self.liquid_aliases

    def _infer_temperature_step(self) -> float:
        if self.temp_step_k is not None:
            return float(self.temp_step_k)

        temps = np.sort(self.gtx_data[self.temp_col].dropna().unique().astype(float))
        if len(temps) < 2:
            return 0.0
        diffs = np.diff(temps)
        diffs = diffs[diffs > 1e-12]
        return float(np.min(diffs)) if len(diffs) else 0.0

    def _infer_grid_delta(self) -> float:
        deltas = []
        for col in self.composition_cols:
            vals = np.sort(self.gtx_data[col].dropna().unique().astype(float))
            if len(vals) < 2:
                continue
            dv = np.diff(vals)
            dv = dv[dv > 1e-12]
            if len(dv):
                deltas.append(float(np.min(dv)))
        return min(deltas) if deltas else 0.0

    @staticmethod
    def _round_sig(value: float, sig: int = 3) -> float:
        """Round a numeric value to a fixed number of significant digits."""
        v = float(value)
        if not np.isfinite(v) or v == 0.0:
            return v
        return float(f"{v:.{sig}g}")

    @staticmethod
    def _connected_components(points: np.ndarray, threshold: float) -> List[List[int]]:
        if len(points) == 0:
            return []

        threshold = max(float(threshold), 0.0)
        if threshold == 0.0:
            return [[i] for i in range(len(points))]

        # Use KD-tree neighborhood queries to avoid dense O(N^2) memory.
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
                nbrs = neighbors[cur]
                for n in nbrs:
                    if not seen[n]:
                        seen[n] = True
                        stack.append(n)
            components.append(sorted(comp))
        return components

    def solve(
        self,
        vertical_simplices: bool = True,
        print_progress: bool = True,
        progress_every: int = 1,
        drop_single_phase_simplices: bool = True,
    ) -> pd.DataFrame:
        """Solve lower-hull equilibrium for each temperature slice.

        Args:
            vertical_simplices: Forwarded to gliq_lowerhull3.
            print_progress: If True, print incremental temperature-slice progress.
            progress_every: Print cadence in number of processed slices.
            drop_single_phase_simplices: If True, discard simplices whose
                vertices all belong to the same phase before storing results.
                This reduces in-memory equilibrium/simplex row accumulation.
        """
        work = self.gtx_data.reset_index(drop=True).copy()
        work['source_row_id'] = np.arange(len(work), dtype=int)

        eq_rows = []
        simplex_rows = []
        simplex_counter = 0
        dropped_single_phase_simplices = 0
        dropped_single_phase_vertices = 0

        grouped = work.groupby(self.temp_col, sort=True)
        self.total_temperature_slices = int(grouped.ngroups)
        self.processed_temperature_slices = []
        self.successful_temperature_slices = []
        self.failed_temperature_slices = []
        self.temperature_slice_status = {}

        progress_every = max(1, int(progress_every))

        if print_progress:
            print(f"[EQ] Starting lower-hull solve over {self.total_temperature_slices} temperature slices")

        for idx, (temp_val, df_t) in enumerate(grouped, start=1):
            t_val = float(temp_val)
            self.processed_temperature_slices.append(t_val)
            df_t = df_t.reset_index(drop=True)
            points = df_t[self.composition_cols + [self.g_col]].to_numpy(dtype=float)

            try:
                simplices = gliq_lowerhull3(points, vertical_simplices=vertical_simplices)
            except QhullError:
                self.failed_temperature_slices.append(t_val)
                self.temperature_slice_status[t_val] = 'qhull_error'
                if print_progress and (idx % progress_every == 0 or idx == self.total_temperature_slices):
                    print(
                        f"[EQ] {idx}/{self.total_temperature_slices} slices | "
                        f"T={t_val:.2f} K | status=qhull_error | "
                        f"success={len(self.successful_temperature_slices)} | "
                        f"failed={len(self.failed_temperature_slices)}"
                    )
                continue

            if simplices is None or len(simplices) == 0:
                self.failed_temperature_slices.append(t_val)
                self.temperature_slice_status[t_val] = 'no_simplices'
                if print_progress and (idx % progress_every == 0 or idx == self.total_temperature_slices):
                    print(
                        f"[EQ] {idx}/{self.total_temperature_slices} slices | "
                        f"T={t_val:.2f} K | status=no_simplices | "
                        f"success={len(self.successful_temperature_slices)} | "
                        f"failed={len(self.failed_temperature_slices)}"
                    )
                continue

            self.successful_temperature_slices.append(t_val)
            self.temperature_slice_status[t_val] = 'ok'

            for simplex in simplices:
                local_idx = [int(i) for i in np.asarray(simplex).tolist()]
                simplex_vertices = df_t.iloc[local_idx].copy()
                source_ids = simplex_vertices['source_row_id'].astype(int).tolist()
                phase_names = simplex_vertices[self.phase_col].astype(str).tolist()

                if drop_single_phase_simplices and len(set(phase_names)) <= 1:
                    dropped_single_phase_simplices += 1
                    dropped_single_phase_vertices += int(len(local_idx))
                    continue

                simplex_id = simplex_counter
                simplex_counter += 1

                simplex_rows.append({
                    'simplex_id': simplex_id,
                    self.temp_col: float(temp_val),
                    'vertex_count': int(len(local_idx)),
                    'vertex_local_indices': local_idx,
                    'vertex_source_row_ids': source_ids,
                    'vertex_phases': phase_names,
                })

                for _, row in simplex_vertices.iterrows():
                    eq_row = {
                        col: float(row[col]) for col in self.composition_cols
                    }
                    eq_row[self.temp_col] = float(row[self.temp_col])
                    if 'T_C' in row.index:
                        eq_row['T_C'] = float(row['T_C'])
                    elif self.temp_col == 'T_K':
                        eq_row['T_C'] = float(row[self.temp_col] - 273.15)
                    else:
                        eq_row['T_C'] = np.nan
                    eq_row['G'] = float(row[self.g_col])
                    eq_row['Phase'] = str(row[self.phase_col])
                    eq_row['simplex_id'] = int(simplex_id)
                    eq_row['source_row_id'] = int(row['source_row_id'])
                    eq_rows.append(eq_row)

            if print_progress and (idx % progress_every == 0 or idx == self.total_temperature_slices):
                print(
                    f"[EQ] {idx}/{self.total_temperature_slices} slices | "
                    f"T={t_val:.2f} K | status=ok | "
                    f"success={len(self.successful_temperature_slices)} | "
                    f"failed={len(self.failed_temperature_slices)}"
                )

        self.equilibrium_df = pd.DataFrame(eq_rows)
        self.simplex_df = pd.DataFrame(simplex_rows)

        if print_progress:
            print(
                f"[EQ] Completed lower-hull sweep: "
                f"processed={len(self.processed_temperature_slices)}/{self.total_temperature_slices}, "
                f"success={len(self.successful_temperature_slices)}, "
                f"failed={len(self.failed_temperature_slices)}, "
                f"dropped_single_phase_simplices={dropped_single_phase_simplices}, "
                f"dropped_single_phase_vertices={dropped_single_phase_vertices}"
            )

        # Lightweight diagnostics for downstream scripts/harnesses.
        self._solve_prune_stats = {
            'drop_single_phase_simplices': bool(drop_single_phase_simplices),
            'dropped_single_phase_simplices': int(dropped_single_phase_simplices),
            'dropped_single_phase_vertices': int(dropped_single_phase_vertices),
            'n_retained_equilibrium_rows': int(len(self.equilibrium_df)),
            'n_retained_simplex_rows': int(len(self.simplex_df)),
        }

        return self.equilibrium_df

    def get_temperature_progress(self) -> Dict[str, Any]:
        """Return tracked per-temperature solve progress for downstream reporting."""
        out = {
            'total_temperature_slices': int(self.total_temperature_slices),
            'n_processed': int(len(self.processed_temperature_slices)),
            'n_successful': int(len(self.successful_temperature_slices)),
            'n_failed': int(len(self.failed_temperature_slices)),
            'processed_temperature_slices': list(self.processed_temperature_slices),
            'successful_temperature_slices': list(self.successful_temperature_slices),
            'failed_temperature_slices': list(self.failed_temperature_slices),
            'temperature_slice_status': dict(self.temperature_slice_status),
        }
        if hasattr(self, '_solve_prune_stats'):
            out['prune_stats'] = dict(self._solve_prune_stats)
        return out

    @staticmethod
    def _cache_file_paths(cache_dir: str, cache_name: str) -> Dict[str, str]:
        base = os.path.join(cache_dir, cache_name)
        return {
            'manifest': base + '_manifest.json',
            'gtx': base + '_gtx.pkl.gz',
            'equilibrium': base + '_equilibrium.pkl.gz',
            'simplex': base + '_simplex.pkl.gz',
        }

    def _build_cache_manifest(self, vertical_simplices: Optional[bool] = None) -> Dict[str, Any]:
        """Build metadata manifest for lower-hull cache artifacts."""
        manifest = {
            'version': 1,
            'n_gtx_rows': int(len(self.gtx_data)),
            'n_equilibrium_rows': int(len(self.equilibrium_df)) if self.equilibrium_df is not None else 0,
            'n_simplex_rows': int(len(self.simplex_df)) if self.simplex_df is not None else 0,
            'composition_cols': list(self.composition_cols),
            'temp_col': self.temp_col,
            'phase_col': self.phase_col,
            'g_col': self.g_col,
            'liquid_aliases': sorted(list(self.liquid_aliases)),
            'cluster_comp_threshold': self.cluster_comp_threshold,
            'temp_step_k': self.temp_step_k,
            'vertical_simplices': vertical_simplices,
        }
        if self.temp_col in self.gtx_data.columns and not self.gtx_data.empty:
            manifest['temp_range'] = [
                float(self.gtx_data[self.temp_col].min()),
                float(self.gtx_data[self.temp_col].max()),
            ]
        return manifest

    def save_lower_hull_cache(
        self,
        cache_dir: str,
        cache_name: str = 'lower_hull',
        include_gtx: bool = True,
        vertical_simplices: Optional[bool] = None,
    ) -> Dict[str, str]:
        """Persist lower-hull solve artifacts to disk.

        Saves equilibrium and simplex dataframes (with simplex_id/source_row_id) and
        optionally the GTX dataframe used as the source for source_row_id mapping.
        """
        if self.equilibrium_df is None or self.simplex_df is None:
            raise ValueError("No lower-hull results to cache. Run solve() first.")

        os.makedirs(cache_dir, exist_ok=True)
        paths = self._cache_file_paths(cache_dir, cache_name)

        self.equilibrium_df.to_pickle(paths['equilibrium'], compression='gzip')
        self.simplex_df.to_pickle(paths['simplex'], compression='gzip')
        if include_gtx:
            self.gtx_data.to_pickle(paths['gtx'], compression='gzip')

        manifest = self._build_cache_manifest(vertical_simplices=vertical_simplices)
        manifest['include_gtx'] = bool(include_gtx)
        with open(paths['manifest'], 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return paths

    def load_lower_hull_cache(
        self,
        cache_dir: str,
        cache_name: str = 'lower_hull',
        replace_gtx_data: bool = True,
        allow_cache_load: bool = False,
    ) -> Dict[str, Any]:
        """Load cached lower-hull artifacts into this solver instance."""
        if not allow_cache_load:
            raise RuntimeError(
                "Cache loading is disabled by default to prevent silent reuse of stale hull data. "
                "If you intentionally want to load cache in a downstream script, call "
                "load_lower_hull_cache(..., allow_cache_load=True)."
            )

        paths = self._cache_file_paths(cache_dir, cache_name)
        required = [paths['manifest'], paths['equilibrium'], paths['simplex']]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing lower-hull cache files: {missing}")

        with open(paths['manifest'], 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        self.equilibrium_df = pd.read_pickle(paths['equilibrium'], compression='gzip')
        self.simplex_df = pd.read_pickle(paths['simplex'], compression='gzip')

        if replace_gtx_data:
            if not os.path.exists(paths['gtx']):
                raise FileNotFoundError(
                    "Cached GTX file is required when replace_gtx_data=True, "
                    f"but not found: {paths['gtx']}"
                )
            self.gtx_data = pd.read_pickle(paths['gtx'], compression='gzip')

        if self.equilibrium_df is not None and not self.equilibrium_df.empty:
            max_source_row_id = int(self.equilibrium_df['source_row_id'].max())
            if max_source_row_id >= len(self.gtx_data):
                raise ValueError(
                    "Cached equilibrium source_row_id exceeds GTX row count. "
                    "Ensure you loaded matching GTX data."
                )

        return manifest

    def solve_with_cache(
        self,
        cache_dir: str,
        cache_name: str = 'lower_hull',
        vertical_simplices: bool = True,
        force_recompute: bool = False,
        include_gtx_in_cache: bool = True,
        use_existing_cache: bool = False,
    ) -> pd.DataFrame:
        """Solve lower hull with optional on-disk cache reuse.

        Default behavior is recompute-first: compute solve() and write cache.
        Cached results are loaded only when use_existing_cache=True and
        force_recompute=False.
        """
        paths = self._cache_file_paths(cache_dir, cache_name)
        cache_exists = all(os.path.exists(paths[k]) for k in ['manifest', 'equilibrium', 'simplex'])

        if use_existing_cache and cache_exists and not force_recompute:
            self.load_lower_hull_cache(
                cache_dir=cache_dir,
                cache_name=cache_name,
                replace_gtx_data=bool(include_gtx_in_cache),
                allow_cache_load=True,
            )
            return self.equilibrium_df

        self.solve(vertical_simplices=vertical_simplices)
        self.save_lower_hull_cache(
            cache_dir=cache_dir,
            cache_name=cache_name,
            include_gtx=include_gtx_in_cache,
            vertical_simplices=vertical_simplices,
        )
        return self.equilibrium_df

    def _coexisting_solids_for_simplex_ids(self, simplex_ids: List[int]) -> List[str]:
        if self.equilibrium_df is None or self.equilibrium_df.empty:
            return []
        mask = self.equilibrium_df['simplex_id'].isin(simplex_ids)
        phases = self.equilibrium_df.loc[mask, 'Phase'].astype(str).tolist()
        solids = sorted({p for p in phases if not self._is_liquid_label(p)})
        return solids

    def get_lowest_liquidus_clusters(
        self,
        comp_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Find cluster-aware lowest-liquidus candidates and simplex-linked solids."""
        if self.equilibrium_df is None:
            self.solve()

        if self.equilibrium_df is None or self.equilibrium_df.empty:
            return {
                'tmin': None,
                'cluster_records': pd.DataFrame(),
                'coexisting_solids_union': [],
            }

        liq_mask = self.equilibrium_df['Phase'].apply(self._is_liquid_label)
        liq_df = self.equilibrium_df.loc[liq_mask].copy()
        if liq_df.empty:
            return {
                'tmin': None,
                'cluster_records': pd.DataFrame(),
                'coexisting_solids_union': [],
            }

        tmin = float(liq_df[self.temp_col].min())
        # Candidate liquid points are taken strictly at the first liquidus
        # temperature (no Tmin + dT window).
        candidates = liq_df[np.isclose(liq_df[self.temp_col], tmin, rtol=1e-12, atol=1e-12)].copy()

        # Keep one representative per composition at Tmin,
        # preferring the lowest-T and then lowest-G row.
        if not candidates.empty:
            candidates = (
                candidates
                .sort_values(by=[self.temp_col, 'G'] + self.composition_cols, ascending=True)
                .drop_duplicates(subset=self.composition_cols, keep='first')
                .reset_index(drop=True)
            )

        if candidates.empty:
            return {
                'tmin': tmin,
                'cluster_records': pd.DataFrame(),
                'coexisting_solids_union': [],
            }

        if comp_threshold is None:
            comp_threshold = self.cluster_comp_threshold
        if comp_threshold is None:
            grid_delta = self._infer_grid_delta()
            comp_threshold = grid_delta if grid_delta > 0 else 0.0

        comp_points = candidates[self.composition_cols].to_numpy(dtype=float)
        components = self._connected_components(comp_points, threshold=float(comp_threshold))

        records = []
        union_solids = set()

        for cluster_id, component in enumerate(components):
            cluster_df = candidates.iloc[component].copy()
            cluster_df = cluster_df.sort_values(
                by=[self.temp_col, 'G'] + self.composition_cols,
                ascending=True
            )
            rep = cluster_df.iloc[0]

            indep_points = cluster_df[self.composition_cols].to_numpy(dtype=float)
            full_points = [
                [self._round_sig(float(1.0 - np.sum(row)))] + [self._round_sig(float(v)) for v in row]
                for row in indep_points
            ]
            indep_points_list = [[self._round_sig(float(v)) for v in row] for row in indep_points]

            simplex_ids = sorted(cluster_df['simplex_id'].astype(int).unique().tolist())
            coexisting_solids = self._coexisting_solids_for_simplex_ids(simplex_ids)
            union_solids.update(coexisting_solids)

            record = {
                'cluster_id': int(cluster_id),
                self.temp_col: float(rep[self.temp_col]),
                'T_C': float(rep['T_C']) if 'T_C' in rep.index and pd.notna(rep['T_C']) else np.nan,
                'G': float(rep['G']),
                'n_points': int(len(cluster_df)),
                'simplex_ids': simplex_ids,
                'coexisting_solids': coexisting_solids,
                'cluster_points_independent': indep_points_list,
                'cluster_points_full': full_points,
            }
            for col in self.composition_cols:
                record[col] = float(rep[col])
            records.append(record)

        records_df = pd.DataFrame(records)
        if not records_df.empty:
            records_df = records_df.sort_values(by=[self.temp_col, 'G'] + self.composition_cols).reset_index(drop=True)

        return {
            'tmin': tmin,
            'cluster_records': records_df,
            'coexisting_solids_union': sorted(union_solids),
        }

    def get_first_interior_liquid_eutectic(
        self,
        interior_tol: float = 1e-6,
        dedupe_by_composition: bool = True,
    ) -> Dict[str, Any]:
        """Generalized counterpart to gliqman_eut_batch first-hit eutectic logic.

        Behavior mirrors the batch script's intent at simplex granularity:
        1) scan equilibrium simplices in ascending temperature,
        2) keep simplices that contain liquid vertices,
        3) compute mean liquid composition within that simplex,
        4) require interior composition (all component fractions > interior_tol),
        5) return first valid candidate.

        Returns:
            dict with eutectic-like temperature/composition and simplex-linked
            phase coexistence info. If none found, fields are None/empty.
        """
        if self.equilibrium_df is None:
            self.solve()

        if self.equilibrium_df is None or self.equilibrium_df.empty:
            return {
                'eutectic_temperature': None,
                'eutectic_temperature_C': None,
                'eutectic_composition_independent': None,
                'eutectic_composition_full': None,
                'simplex_ids': [],
                'coexisting_solids': [],
                'n_liquid_points_at_temperature': 0,
            }

        if self.simplex_df is None or self.simplex_df.empty:
            return {
                'eutectic_temperature': None,
                'eutectic_temperature_C': None,
                'eutectic_composition_independent': None,
                'eutectic_composition_full': None,
                'simplex_ids': [],
                'coexisting_solids': [],
                'n_liquid_points_at_temperature': 0,
            }

        simplex_scan = self.simplex_df.sort_values(by=[self.temp_col, 'simplex_id'], ascending=True)

        for _, simplex_meta in simplex_scan.iterrows():
            simplex_id = int(simplex_meta['simplex_id'])
            simplex_rows = self.equilibrium_df[self.equilibrium_df['simplex_id'] == simplex_id].copy()
            if simplex_rows.empty:
                continue

            liq_t = simplex_rows[simplex_rows['Phase'].apply(self._is_liquid_label)].copy()
            if liq_t.empty:
                continue

            if dedupe_by_composition:
                liq_t = (
                    liq_t
                    .sort_values(by=['G'] + self.composition_cols, ascending=True)
                    .drop_duplicates(subset=self.composition_cols, keep='first')
                    .reset_index(drop=True)
                )

            indep_mean = liq_t[self.composition_cols].mean(axis=0)
            indep_vals_raw = [float(indep_mean[col]) for col in self.composition_cols]
            indep_vals = [self._round_sig(v) for v in indep_vals_raw]
            dep_val_raw = float(1.0 - np.sum(indep_vals_raw))
            dep_val = self._round_sig(dep_val_raw)
            full_comp = [dep_val] + indep_vals

            if not all(v > interior_tol for v in [dep_val_raw] + indep_vals_raw):
                continue

            simplex_ids = [simplex_id]
            coexisting_solids = sorted({
                p for p in simplex_rows['Phase'].astype(str).unique().tolist()
                if not self._is_liquid_label(p)
            })

            temp_val = float(simplex_rows[self.temp_col].iloc[0])

            return {
                'eutectic_temperature': float(temp_val),
                'eutectic_temperature_C': float(temp_val - 273.15) if self.temp_col == 'T_K' else np.nan,
                'eutectic_composition_independent': indep_vals,
                'eutectic_composition_full': full_comp,
                'simplex_ids': simplex_ids,
                'coexisting_solids': coexisting_solids,
                'n_liquid_points_at_temperature': int(len(liq_t)),
            }

        return {
            'eutectic_temperature': None,
            'eutectic_temperature_C': None,
            'eutectic_composition_independent': None,
            'eutectic_composition_full': None,
            'simplex_ids': [],
            'coexisting_solids': [],
            'n_liquid_points_at_temperature': 0,
        }






if __name__ == "__main__":
    # Validation-only harness (no plotting):
    # - uses real L-params from existing Excel path
    # - checks canonical ordering + L1 sign normalization
    # - checks GTX schema and temperature consistency
    # - checks same-composition multi-phase readiness from solid polymorph refs

    def _load_binary_params_from_file(df: pd.DataFrame, pair_label: str) -> list[float]:
        """Load [L0_a, L0_b, L1_a, L1_b] for a canonical pair from dataframe.

        If only reversed ordering is found in the file, flip L1 signs to map
        into canonical sorted pair convention.
        """
        canonical = "-".join(sorted(pair_label.split('-')))
        parts = canonical.split('-')
        reversed_label = f"{parts[1]}-{parts[0]}"

        if canonical in df['system'].tolist():
            row = df[df['system'] == canonical].iloc[0]
            l0_a = float(row['L0_a'])
            l0_b = float(row['L0_b'])
            l1_a = float(row['L1_a'])
            l1_b = float(row['L1_b'])
            return [l0_a, l0_b, l1_a, l1_b]

        if reversed_label in df['system'].tolist():
            row = df[df['system'] == reversed_label].iloc[0]
            l0_a = float(row['L0_a'])
            l0_b = float(row['L0_b'])
            l1_a = -float(row['L1_a'])
            l1_b = -float(row['L1_b'])
            return [l0_a, l0_b, l1_a, l1_b]

        raise ValueError(f"Binary pair '{pair_label}' not found in parameter file.")

    print("=" * 72)
    print("GENERAL GTX VALIDATION HARNESS (NO PLOTTING)")
    print("=" * 72)

    # Intentionally unsorted input to verify canonical remapping behavior with reference phases that have polymorphs (e.g. elemental refs for Zr and Y).
    # input_elements = ["Zr", "Al", "Y", "Fe"]
    # input_elements = ["Bi", "Cd", "Sn", "Ag"]
    # input_elements = ["Bi", "Cd", "Sn"]
    # input_elements = ["Al", "Cu", "Si", "Mg"]
    # input_elements = ["Pb", "Sn", "Cd", "Zn"]
    # input_elements = ["Pb", "Sn", "Cd", "Bi"]
    # input_elements = ["Ag", "Sn", "Bi", "Zn"]
    
    input_elements = ["Zr", "Hf", "Nb", "W"] # SS only

    include_polymorphs = False # Toggle False for MP-only reference mode.
    include_solid_solutions = True # Toggle True to enable reference solid-solution cloud generation (BCC/FCC/HCP).
    use_all_temps_for_equilibrium_validation = True# True -> all T slices; False -> low/mid/high only.
    canonical_elements = sorted(input_elements)
    binary_pairs = [
        f"{canonical_elements[i]}-{canonical_elements[j]}"
        for i, j in combinations(range(len(canonical_elements)), 2)
    ]

    print(f"Input elements: {input_elements}")
    print(f"Canonical elements: {canonical_elements}")
    print(f"Canonical binary pairs: {binary_pairs}")
    print(f"Polymorph reference mode: {include_polymorphs}")
    print(f"Equilibrium validation uses all temperatures: {use_all_temps_for_equilibrium_validation}")

    # param_file = "data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx" # for general runs
    param_file = "data/high_component/ssol_fits_linear_model_legacy_refs-tau_penalty.xlsx" # for solid-solution reference mode
    print(f"Loading binary parameters from: {param_file}")
    binary_param_df = pd.read_excel(param_file)
    print(binary_param_df)

    binary_L_dict: Dict[str, List[float]] = {
        pair: _load_binary_params_from_file(binary_param_df, pair)
        for pair in binary_pairs
    }

    output_dir = "all_dumps/quaternary_demo/"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Validation 1: Canonical ordering + sign normalization
    # -----------------------------------------------------------------------
    interp = GeneralInterpolation(
        elements=input_elements,
        output_dir=output_dir,
        grid_delta=0.025,
        include_polymorphs=include_polymorphs,
        include_ref_solid_solutions=include_solid_solutions,
        temp_delta_k=10.0,
    )
    interp.set_binary_params(binary_L_dict)

    assert interp.elements == canonical_elements, (
        f"Canonical element ordering failed: {interp.elements} != {canonical_elements}"
    )
    # assert interp.param_format == 'combined', f"Expected param_format='combined', got {interp.param_format}"

    # Explicit sign-normalization check using real loaded params from file.
    check_pair = binary_pairs[0]
    p0, p1 = check_pair.split('-')
    reversed_key = f"{p1}-{p0}"
    mixed_dict = dict(binary_L_dict)
    l0_a, l0_b, l1_a, l1_b = mixed_dict.pop(check_pair)
    mixed_dict[reversed_key] = [l0_a, l0_b, -l1_a, -l1_b]

    interp_mixed = GeneralInterpolation(
        elements=input_elements,
        output_dir=output_dir,
        include_polymorphs=include_polymorphs,
        include_ref_solid_solutions=include_solid_solutions,
        temp_delta_k=10.0,
    )
    interp_mixed.set_binary_params(mixed_dict)
    norm = interp_mixed.binary_L_params[check_pair]
    ref = binary_L_dict[check_pair]
    assert np.allclose(norm, ref, rtol=1e-12, atol=1e-12), (
        f"L1 sign normalization mismatch for {check_pair}: normalized={norm}, expected={ref}"
    )
    print("[PASS] Canonical element ordering and binary sign normalization")

    # -----------------------------------------------------------------------
    # Validation 2: Full interpolation run and GTX schema checks
    # -----------------------------------------------------------------------
    interp.interpolate(use_mp_cache=True)
    gtx = interp.gtx_data

    required_cols = set(interp.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase Name'])
    missing_cols = required_cols - set(gtx.columns)
    assert not missing_cols, f"Missing required GTX columns: {missing_cols}"

    assert np.isfinite(gtx['T_K']).all(), "Non-finite temperatures found in T_K"
    assert (gtx['T_K'] >= 0.0).all(), "Found temperatures below 0 K"
    assert np.allclose(gtx['T_C'].to_numpy(), gtx['T_K'].to_numpy() - 273.15, rtol=1e-12, atol=1e-9), (
        "T_C column is inconsistent with T_K - 273.15"
    )

    comp_cols = interp.get_composition_columns()
    dep = 1.0 - gtx[comp_cols].sum(axis=1)
    assert ((dep >= -1e-8) & (dep <= 1.0 + 1e-8)).all(), "Dependent composition is out of physical bounds"

    print("[PASS] GTX schema and temperature/composition sanity checks")
    print(f"       GTX rows: {len(gtx)}")
    print(f"       T range: {gtx['T_K'].min():.2f} K to {gtx['T_K'].max():.2f} K")

    # -----------------------------------------------------------------------
    # Validation 3: Polymorph readiness (same composition, multiple solid phases)
    # -----------------------------------------------------------------------
    solid_gtx = interp.solid_gtx if interp.solid_gtx is not None else interp.compute_solid_gtx()
    solid_group = solid_gtx.groupby(comp_cols)['Phase Name'].nunique().reset_index(name='n_phase')
    same_comp_multi = int((solid_group['n_phase'] > 1).sum())

    print(f"[INFO] Same-composition multi-solid reference points: {same_comp_multi}")
    print(f"[INFO] Elemental polymorph refs added: {interp.metadata.get('n_elemental_polymorph_refs', 0)}")
    if same_comp_multi == 0:
        print("[WARN] No same-composition multi-phase solids found for this system.")

    # -----------------------------------------------------------------------
    # Validation 4: GeneralEquilibrium solve + cluster-aware invariant utility
    # -----------------------------------------------------------------------
    # Toggle between full-temperature equilibrium validation and a compact
    # representative subset (low/mid/high) for faster iteration.
    unique_temps = sorted(gtx['T_K'].unique())
    if use_all_temps_for_equilibrium_validation:
        selected_temps = unique_temps
    else:
        if len(unique_temps) >= 3:
            selected_temps = [
                unique_temps[0],
                unique_temps[len(unique_temps) // 2],
                unique_temps[-1],
            ]
        else:
            selected_temps = unique_temps
    gtx_subset = gtx[gtx['T_K'].isin(selected_temps)].reset_index(drop=True)
    print(f"[INFO] Equilibrium validation temperatures: {selected_temps}")
    print(f"[INFO] Input subset liquid rows: {(gtx_subset['Phase Name'] == 'L').sum()}")

    eq_solver = GeneralEquilibrium(gtx_subset)
    eq_df = eq_solver.solve(vertical_simplices=True)

    # Persist lower-hull artifacts for downstream visualization/post-processing.
    # Keep this separate from MP cache to avoid accidental coupling.
    lower_hull_cache_dir = os.path.join(output_dir, "lower_hull_cache")
    os.makedirs(lower_hull_cache_dir, exist_ok=True)
    lower_hull_cache_name = f"{'-'.join(canonical_elements)}_T{selected_temps[0]:.2f}_{selected_temps[-1]:.2f}"
    cache_paths = eq_solver.save_lower_hull_cache(
        cache_dir=lower_hull_cache_dir,
        cache_name=lower_hull_cache_name,
        include_gtx=True,
        vertical_simplices=True,
    )
    print(f"[INFO] Saved lower-hull cache manifest: {cache_paths['manifest']}")
    print(f"[INFO] Saved lower-hull equilibrium cache: {cache_paths['equilibrium']}")
    print(f"[INFO] Saved lower-hull simplex cache: {cache_paths['simplex']}")
    print(f"[INFO] Saved lower-hull GTX cache: {cache_paths['gtx']}")

    assert not eq_df.empty, "GeneralEquilibrium.solve returned no rows"
    eq_required_cols = set(interp.get_composition_columns() + ['T_K', 'T_C', 'G', 'Phase', 'simplex_id'])
    assert eq_required_cols.issubset(eq_df.columns), (
        f"GeneralEquilibrium output missing columns: {eq_required_cols - set(eq_df.columns)}"
    )
    assert eq_df['simplex_id'].is_unique is False, "Expected repeated simplex IDs across simplex vertices"

    inv = eq_solver.get_lowest_liquidus_clusters()
    assert 'cluster_records' in inv and 'coexisting_solids_union' in inv, (
        "Invariant utility output missing expected keys"
    )

    eut_like = eq_solver.get_first_interior_liquid_eutectic(interior_tol=1e-6)
    print(f"[PASS] GeneralEquilibrium solve rows: {len(eq_df)}")
    print(f"[INFO] Equilibrium liquid rows: {(eq_df['Phase'] == 'L').sum()}")
    print(f"[PASS] Lowest-liquidus clusters found: {len(inv['cluster_records'])}")
    if len(inv['cluster_records']) > 0:
        cluster_temps = inv['cluster_records']['T_K'].to_numpy(dtype=float)
        same_temp = np.all(np.isclose(cluster_temps, cluster_temps[0], rtol=1e-12, atol=1e-12))
        print(f"[INFO] All lowest-liquidus clusters share the same temperature: {same_temp}")
        for _, row in inv['cluster_records'].iterrows():
            cid = int(row['cluster_id'])
            print(f"[INFO] Cluster {cid} full compositions: {row['cluster_points_full']}")
    print(f"[INFO] Coexisting solids union: {inv['coexisting_solids_union']}")
    print(f"[INFO] First-interior-liquid eutectic-like result: {eut_like}")

    # Save compact validation snapshot.
    snapshot_path = os.path.join(output_dir, "gtx_validation_snapshot.csv")
    gtx.head(5000).to_csv(snapshot_path, index=False)
    print(f"Saved validation snapshot: {snapshot_path}")

    print("=" * 72)
    print("VALIDATION HARNESS COMPLETE")
    print("=" * 72)

    print(eq_df)

    # print rows of eq_df with Phase = 'L' 
    print("\nEquilibrium rows with Phase = 'L':")
    print(eq_df[eq_df['Phase'] == 'L'])

    print(gtx)
    print(solid_gtx)
    print(solid_group)