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
from scipy.spatial import Delaunay
from copy import deepcopy

from gliquid.config import fusion_enthalpies_file, fusion_temps_file
from gliquid.binary import (
    BinaryLiquid,
    linear_expr, exponential_expr, combined_expr)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # If importing this file into a script from a different dir
from extensive_hull_main import gliq_lowerhull3, gen_hyperplane_eqns2_optimized
import random
from auth import mpapi_key

mpr = MPRester(mpapi_key)  

R = 8.314  # J/(mol*K)


# ==============================================================================
# SYMBOLIC VARIABLE MANAGEMENT
# ==============================================================================

from typing import Dict, Tuple, Optional, Callable
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
        x: Tuple of independent mole fraction symbols.
        mole_fractions: Tuple of all mole fractions (including dependent one).
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
        
        # Mole fraction symbols: x0, x1, ..., x[n-2]
        self.x = sp.symbols(f'x0:{self.n_components - 1}', real=True, nonnegative=True)
        if self.n_components == 2:
            self.x = (self.x,) if not isinstance(self.x, tuple) else self.x
        
        # All mole fractions including the dependent one
        x_indep = list(self.x)
        x_dep = 1 - sum(self.x)
        self.mole_fractions = tuple(x_indep + [x_dep])
        
        # Binary pairs
        self.binary_pairs = list(combinations(range(self.n_components), 2))
    
    @property
    def n_binary_pairs(self) -> int:
        """Number of binary pairs: n*(n-1)/2"""
        return len(self.binary_pairs)
    
    def get_mole_fraction(self, index: int) -> sp.Expr:
        """Get mole fraction expression for component at given index."""
        if index < 0 or index >= self.n_components:
            raise IndexError(f"Component index {index} out of range [0, {self.n_components - 1}]")
        return self.mole_fractions[index]
    
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
        for xi in self.mole_fractions:
            term = sp.Piecewise(
                (0, sp.Eq(xi, 0)),
                (xi * sp.log(xi), True)
            )
            mixing_terms.append(term)
        return R * self.t * sum(mixing_terms)
    
    def _build_reference_gibbs(self) -> sp.Expr:
        """Build reference Gibbs energy: G_ref = sum(xi * Gi_ref)."""
        return sum(
            self.mole_fractions[i] * self.g_ref_exprs[i]
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
        weight, xi_eff, xj_eff = weight_func(self.mole_fractions, i, j)
        
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
                product_term *= self.mole_fractions[idx]
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
    parameters to compute thermodynamic HSX data over a composition grid.
    Also retrieves DFT formation energies of solid phases from Materials Project.
    
    This class uses ThermoExprBuilder internally for symbolic expression building
    but does not inherit from it - it's a composition-based design.
    
    Attributes:
        elements: List of element symbols in sorted alphabetical order.
        n_components: Number of components in the system.
        output_dir: Directory for caching MP data and outputs.
        binary_pairs: List of binary pair strings (e.g., ["Al-Cu", "Al-Mg"]).
        hsx_data: DataFrame containing the final HSX dataset.
        
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
        >>> df = interp.hsx_data
    """
    
    def __init__(
        self,
        elements: List[str],
        output_dir: str,
        grid_delta: float = 0.025,
        param_format: str = 'linear',
        interp_scheme: str = 'linear',
        tau: float = 8000,
        mp_api_key: Optional[str] = None
    ):
        """
        Initialize the interpolation system.
        
        Args:
            elements: List of element symbols (e.g., ['Al', 'Cu', 'Si', 'Mg']).
            output_dir: Directory for caching data and saving outputs.
            grid_delta: Composition grid spacing (default 0.025).
            param_format: L parameter format ('linear', 'exponential', 'combined').
            interp_scheme: Interpolation scheme ('linear', 'muggianu', 'kohler').
            tau: Time constant for combined format (default 8000).
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
        self.param_format = param_format
        self.interp_scheme = interp_scheme
        self.tau = tau
        
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
        self.liquid_hsx: Optional[pd.DataFrame] = None
        self.solid_hsx: Optional[pd.DataFrame] = None
        self.hsx_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, any] = {}
        
        # Internal builder (created during interpolation)
        self._expr_builder: Optional[ThermoExprBuilder] = None
        
        print(f"Initialized GeneralInterpolation for system: {'-'.join(self.elements)}")
        print(f"  Components: {self.n_components}")
        print(f"  Binary pairs: {self.binary_pairs}")
    
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
        Load fusion enthalpies and temperatures for all elements from config files.
        
        Returns:
            self (for method chaining)
        """
        with open(fusion_enthalpies_file) as f:
            all_enthalpies = json.load(f)
        with open(fusion_temps_file) as f:
            all_temps = json.load(f)
        
        self.fusion_data = {
            'enthalpies': [all_enthalpies[el] for el in self.elements],
            'temperatures': [all_temps[el] for el in self.elements],
        }
        
        # Compute entropies: S_fus = H_fus / T_fus
        self.fusion_data['entropies'] = [
            h / t for h, t in zip(
                self.fusion_data['enthalpies'], 
                self.fusion_data['temperatures']
            )
        ]
        
        return self
    
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
        # Normalize keys to alphabetical order
        normalized_params = {}
        for key, params in L_params.items():
            parts = key.split('-')
            normalized_key = '-'.join(sorted(parts))
            normalized_params[normalized_key] = params
        
        # Check all pairs are provided
        missing = set(self.binary_pairs) - set(normalized_params.keys())
        if missing:
            raise ValueError(f"Missing L parameters for binary pairs: {missing}")
        
        self.binary_L_params = normalized_params
        self.binary_fit_types = fit_types or {pair: 'pred' for pair in self.binary_pairs}
        
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
            params: [H_param, S_param] or similar format.
        
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
        Generate an n-dimensional composition grid where sum(x_i) = 1.
        
        Returns:
            Array of shape (n_points, n_components) with valid compositions.
        """
        # Create 1D grid
        steps = np.arange(0, 1 + self.grid_delta, self.grid_delta)
        
        # Generate all combinations
        grids = np.meshgrid(*[steps] * self.n_components)
        compositions = np.stack([g.flatten() for g in grids], axis=1)
        
        # Filter to valid compositions (sum ≈ 1)
        sums = compositions.sum(axis=1)
        valid_mask = np.isclose(sums, 1.0, atol=1e-6)
        valid_compositions = compositions[valid_mask]
        
        # Round to avoid floating point issues
        decimal_places = max(2, -int(np.log10(self.grid_delta)))
        valid_compositions = np.round(valid_compositions, decimal_places)
        
        return valid_compositions
    
    # =========================================================================
    # Liquid HSX Computation
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
        
        # Set reference Gibbs energies
        builder.set_reference_gibbs_from_fusion(
            self.fusion_data['enthalpies'],
            self.fusion_data['temperatures']
        )
        
        # Convert string-keyed params to index-keyed params
        if self.binary_L_params is not None:
            index_params = {
                self._pair_str_to_idx[pair]: params
                for pair, params in self.binary_L_params.items()
            }
            builder.set_binary_L_params(index_params)
        
        # Add higher-order terms
        for indices, params in self.higher_order_params.items():
            # Assume params = [H, S] -> expression = H - T*S
            expr = params[0] - builder.t * params[1] if len(params) > 1 else params[0]
            builder.add_higher_order_term(indices, expr)
        
        builder.build()
        return builder
    
    def compute_liquid_hsx(self) -> pd.DataFrame:
        """
        Compute liquid phase HSX data on the composition grid.
        
        Returns:
            DataFrame with columns: x0, x1, ..., x[n-2], H, S, 'Phase Name'
        """
        # Build expressions
        self._expr_builder = self._build_expressions()
        
        # Generate composition grid
        compositions = self._generate_composition_grid()
        
        # Get lambdified functions
        h_func = self._expr_builder.lambdify('h_liquid')
        s_func = self._expr_builder.lambdify('s_liquid')
        
        # Prepare evaluation arguments
        # T = mean melting temperature
        mean_temp = np.mean(self.fusion_data['temperatures'])
        
        # Independent compositions: x0, x1, ..., x[n-2]
        x_indep = [compositions[:, i] for i in range(self.n_components - 1)]
        eval_args = x_indep + [mean_temp]
        
        # Evaluate with error suppression for edge cases
        with np.errstate(divide='ignore', invalid='ignore'):
            h_values = h_func(*eval_args)
            s_values = s_func(*eval_args)
        
        # Replace inf/nan with 0
        h_values = np.where(np.isfinite(h_values), h_values, 0)
        s_values = np.where(np.isfinite(s_values), s_values, 0)
        
        # Build DataFrame
        data = {}
        for i in range(self.n_components - 1):
            data[f'x{i}'] = compositions[:, i]
        data['H'] = h_values
        data['S'] = s_values
        data['Phase Name'] = 'L'
        
        self.liquid_hsx = pd.DataFrame(data)
        return self.liquid_hsx
    
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
        
        element_objects = [Element(el) for el in self.elements]
        
        for entry in stable_entries:
            formula = entry.composition.reduced_formula
            comp = Composition(formula)
            entry_elements = comp.elements
            
            # Formation energy in J/mol (converted from eV/atom)
            form_energy = phase_diagram.get_form_energy_per_atom(entry) * 96485
            formation_energies.append(form_energy)
            
            # Get atomic fractions for each element
            atom_fracs = []
            for element in element_objects:
                if element in entry_elements:
                    atom_fracs.append(comp.get_atomic_fraction(element))
                else:
                    atom_fracs.append(0.0)
            
            compositions_list.append(atom_fracs)
            phase_names.append(formula)
        
        compositions_arr = np.array(compositions_list)
        
        # Track deepest formation energy
        self.metadata['deepest_formation_energy'] = min(formation_energies)
        
        # Build DataFrame (exclude last composition column since it's dependent)
        data = {}
        for i in range(self.n_components - 1):
            data[f'x{i}'] = compositions_arr[:, i]
        data['H'] = formation_energies
        data['S'] = [0.0] * len(formation_energies)  # Solids have S=0 in this model
        data['Phase Name'] = phase_names
        
        df = pd.DataFrame(data)
        
        # Keep only lowest energy entry for each phase
        df = df.loc[df.groupby('Phase Name')['H'].idxmin()]
        df = df.reset_index(drop=True)
        
        self.solid_hsx = df
        return self.solid_hsx
    
    # =========================================================================
    # Main Interpolation Pipeline
    # =========================================================================
    
    def interpolate(self, use_mp_cache: bool = True) -> 'GeneralInterpolation':
        """
        Run the full interpolation pipeline.
        
        This method:
        1. Loads fusion data (if not already loaded)
        2. Computes liquid phase HSX on composition grid
        3. Fetches solid formation energies from Materials Project
        4. Combines into final HSX dataset
        
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
        
        # Step 2: Compute liquid HSX
        print("\n[2/3] Computing liquid phase HSX...")
        self.compute_liquid_hsx()
        print(f"       Generated {len(self.liquid_hsx)} liquid phase points")
        
        # Step 3: Fetch solid formation energies
        print("\n[3/3] Fetching solid formation energies...")
        self.fetch_solid_formation_energies(use_cache=use_mp_cache)
        print(f"       Found {len(self.solid_hsx)} stable solid phases")
        
        # Combine dataframes
        self.hsx_data = pd.concat(
            [self.liquid_hsx, self.solid_hsx], 
            ignore_index=True
        )
        self.hsx_data = self.hsx_data.drop_duplicates()
        self.hsx_data = self.hsx_data.reset_index(drop=True)
        
        print(f"\nInterpolation complete!")
        print(f"  Total HSX points: {len(self.hsx_data)}")
        print(f"  Phases: {self.hsx_data['Phase Name'].nunique()}")
        
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
    
    def save_hsx(self, filename: Optional[str] = None) -> str:
        """
        Save the HSX data to a CSV file.
        
        Args:
            filename: Output filename. If None, uses system name.
        
        Returns:
            Path to saved file.
        """
        if self.hsx_data is None:
            raise ValueError("No HSX data to save. Run interpolate() first.")
        
        if filename is None:
            filename = f"{'-'.join(self.elements)}_hsx.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        self.hsx_data.to_csv(filepath, index=False)
        print(f"Saved HSX data to: {filepath}")
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
            'fusion_data_loaded': self.fusion_data is not None,
            'binary_params_set': self.binary_L_params is not None,
            'n_liquid_points': len(self.liquid_hsx) if self.liquid_hsx is not None else 0,
            'n_solid_phases': len(self.solid_hsx) if self.solid_hsx is not None else 0,
            'total_hsx_points': len(self.hsx_data) if self.hsx_data is not None else 0,
            **self.metadata
        }


# ==============================================================================
# GENERAL PLOTTER CLASS (T-loop approach following ternary_HSX.py)
# ==============================================================================

def ternary_to_cartesian(x0, x1):
    """Convert ternary coordinates to cartesian for plotting."""
    cart_x = x0 + 0.5 * x1
    cart_y = np.sqrt(3) / 2 * x1
    return cart_x, cart_y


def cartesian_to_ternary_df(df):
    """Convert ternary compositions to cartesian coordinates in a DataFrame."""
    df = df.copy()
    cart_coords = [ternary_to_cartesian(row['x0'], row['x1']) for _, row in df.iterrows()]
    df['x0'] = [c[0] for c in cart_coords]
    df['x1'] = [c[1] for c in cart_coords]
    return df


class GeneralPlotter:
    """
    General n-component phase diagram plotter for ternary slices.
    
    Follows the ternary_HSX.py approach:
    - Loop over temperatures T
    - For each T, compute G = H - T*S
    - Compute 2D convex hull in (x0, x1, G) space
    - Collect simplices and phase labels
    
    Attributes:
        hsx_data: DataFrame with columns [x0, x1, ..., S, H, 'Phase Name'].
        elements: List of element symbols in order matching x0, x1, ...
        n_components: Number of components.
        
    Example:
        >>> plotter = GeneralPlotter(hsx_data, elements=['Al', 'Ba', 'Mg', 'Si'])
        >>> plotter.define_ternary_slice(fixed_element='Al', fixed_value=0.0)
        >>> plotter.process_data()
        >>> fig = plotter.plot_ternary()
    """
    
    def __init__(
        self,
        hsx_data: pd.DataFrame,
        elements: List[str],
        output_dir: str = '.',
        T_incr: float = 10.0,
        temp_slider: List[float] = None,
        T_min_filter: float = -200.0,  # Minimum temperature filter (°C)
        T_max_filter: float = 2500.0,  # Maximum temperature filter (°C)
        tri_T_span_max: float = 500.0,  # Maximum T span for a triangle (°C)
    ):
        """
        Initialize the plotter.
        
        Args:
            hsx_data: DataFrame with columns [x0, x1, ..., S, H, 'Phase Name'].
            elements: List of element symbols in order matching composition columns.
            output_dir: Directory for outputs.
            T_incr: Temperature increment for the T grid (default: 10K).
            temp_slider: [lower_extension, upper_extension] for temperature range.
            T_min_filter: Minimum temperature (°C) for liquidus points (filters vertical simplices).
            T_max_filter: Maximum temperature (°C) for liquidus points.
            tri_T_span_max: Maximum temperature span (°C) for a valid triangle.
        """
        self.hsx_data = hsx_data.copy()
        self.elements = elements
        self.n_components = len(elements)
        self.output_dir = output_dir
        self.T_incr = T_incr
        self.temp_slider = temp_slider if temp_slider is not None else [0, 0]
        self.T_min_filter = T_min_filter
        self.T_max_filter = T_max_filter
        self.tri_T_span_max = tri_T_span_max
        
        # Validate input
        comp_cols = [f'x{i}' for i in range(self.n_components - 1)]
        required_cols = comp_cols + ['S', 'H', 'Phase Name']
        missing = set(required_cols) - set(self.hsx_data.columns)
        if missing:
            raise ValueError(f"HSX data missing required columns: {missing}")
        
        # Slice definition
        self._slice_defined = False
        self._fixed_element: Optional[str] = None
        self._fixed_value: float = 0.0
        self._slice_elements: List[str] = []
        self._slice_hsx: Optional[pd.DataFrame] = None
        
        # Processing state
        self.equil_df_list: List[pd.DataFrame] = []
        self.T_grid: Optional[np.ndarray] = None
        self.conds: Optional[List[float]] = None
        self.df_Tgroups: Dict[float, pd.DataFrame] = {}
        
        # Plotting data
        self.plotting_df: Optional[pd.DataFrame] = None
        self.liq_plotting_df: Optional[pd.DataFrame] = None
        self.solid_plotting_df: Optional[pd.DataFrame] = None
        
        # Color mapping
        self._init_color_map()
        
        print(f"Initialized GeneralPlotter for {self.n_components}-component system")
        print(f"  Elements: {self.elements}")
        print(f"  HSX points: {len(self.hsx_data)}")
    
    def _init_color_map(self):
        """Initialize color mapping for phases."""
        phases = self.hsx_data['Phase Name'].unique()
        solid_phases = [p for p in phases if p != 'L']
        
        color_array = px.colors.qualitative.Vivid
        color_array = color_array * (len(solid_phases) // len(color_array) + 1)
        
        self.color_map = {phase: color for phase, color in zip(solid_phases, color_array)}
        self.color_map['L'] = 'cornflowerblue'
    
    # =========================================================================
    # Slice Definition
    # =========================================================================
    
    def define_ternary_slice(
        self,
        fixed_element: str,
        fixed_value: float = 0.0,
        tolerance: float = 0.01
    ) -> 'GeneralPlotter':
        """
        Define a ternary slice by fixing one element's composition.
        
        Args:
            fixed_element: Element symbol to fix (e.g., 'Al').
            fixed_value: Composition value for the fixed element (0.0 to 1.0).
            tolerance: Tolerance for filtering compositions near fixed_value.
        
        Returns:
            self (for method chaining)
        """
        if fixed_element not in self.elements:
            raise ValueError(f"Unknown element '{fixed_element}'. Available: {self.elements}")
        
        if not (0.0 <= fixed_value <= 1.0):
            raise ValueError(f"fixed_value must be between 0 and 1, got {fixed_value}")
        
        self._fixed_element = fixed_element
        self._fixed_value = fixed_value
        
        # Determine slice elements (the 3 elements that form the ternary)
        self._slice_elements = [el for el in self.elements if el != fixed_element]
        if len(self._slice_elements) != 3:
            raise ValueError(f"Expected 3 remaining elements for ternary slice, got {len(self._slice_elements)}")
        
        # Get the index of the fixed element
        fixed_idx = self.elements.index(fixed_element)
        
        # Filter HSX data for compositions near the fixed value
        if fixed_idx < self.n_components - 1:
            # Fixed element has direct column (x0, x1, x2, ...)
            col_name = f'x{fixed_idx}'
            mask = np.abs(self.hsx_data[col_name] - fixed_value) < tolerance
        else:
            # Fixed element is the dependent variable (1 - sum(x_i))
            comp_cols = [f'x{i}' for i in range(self.n_components - 1)]
            sum_others = self.hsx_data[comp_cols].sum(axis=1)
            dependent_val = 1.0 - sum_others
            mask = np.abs(dependent_val - fixed_value) < tolerance
        
        self._slice_hsx = self.hsx_data[mask].copy()
        
        if len(self._slice_hsx) == 0:
            raise ValueError(f"No data points found for {fixed_element}={fixed_value} (tolerance={tolerance})")
        
        # Renormalize compositions for the ternary slice
        # Map the remaining elements to x0, x1 (2 independent variables for ternary)
        self._renormalize_slice_compositions(fixed_idx)
        
        self._slice_defined = True
        
        print(f"Defined ternary slice: {fixed_element} = {fixed_value*100:.1f}%")
        print(f"  Slice elements: {self._slice_elements}")
        print(f"  Slice points: {len(self._slice_hsx)}")
        
        return self
    
    def _renormalize_slice_compositions(self, fixed_idx: int):
        """Renormalize compositions for the ternary slice.
        
        Following ternary_HSX.py convention where for sorted elements [A, B, C]:
        - x0 = B composition (2nd element)
        - x1 = C composition (3rd element)  
        - dependent = A composition (1st element = 1 - x0 - x1)
        """
        # Get composition values for the 3 slice elements (in sorted order)
        slice_comps = []
        for i, el in enumerate(self.elements):
            if el == self._fixed_element:
                continue
            el_idx = self.elements.index(el)
            if el_idx < self.n_components - 1:
                slice_comps.append(self._slice_hsx[f'x{el_idx}'].values)
            else:
                comp_cols = [f'x{j}' for j in range(self.n_components - 1)]
                slice_comps.append((1.0 - self._slice_hsx[comp_cols].sum(axis=1)).values)
        
        # Renormalize so they sum to 1 within the slice
        slice_comps = np.array(slice_comps).T  # Shape: (n_points, 3)
        row_sums = slice_comps.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        slice_comps = slice_comps / row_sums
        
        # Match ternary_HSX.py convention: x0=B (2nd element), x1=C (3rd element), A=dependent
        # slice_comps columns: [0]=A (1st slice element), [1]=B (2nd), [2]=C (3rd)
        self._slice_hsx['x0'] = slice_comps[:, 1]  # B = 2nd slice element
        self._slice_hsx['x1'] = slice_comps[:, 2]  # C = 3rd slice element
        # A = 1 - x0 - x1 (1st slice element is now dependent, matching ternary_HSX.py)
        
        # Round to avoid floating point issues
        self._slice_hsx['x0'] = self._slice_hsx['x0'].round(4)
        self._slice_hsx['x1'] = self._slice_hsx['x1'].round(4)
        self._slice_hsx = self._slice_hsx.rename(columns={'Phase Name': 'Phase'})
        self._slice_hsx['Colors'] = self._slice_hsx['Phase'].map(self.color_map)
    
    # =========================================================================
    # Data Processing (T-loop approach from ternary_HSX.py)
    # =========================================================================
    
    def _init_sys(self):
        """Initialize system parameters for T-loop processing."""
        if not self._slice_defined:
            raise RuntimeError("Call define_ternary_slice() first")
        
        # Estimate temperature range from fusion data
        # Use H/S ratio as proxy for melting temperature
        liquid_df = self._slice_hsx[self._slice_hsx['Phase'] == 'L']
        if len(liquid_df) > 0:
            valid_mask = liquid_df['S'] > 0.1  # Avoid division by zero
            if valid_mask.sum() > 0:
                temps = liquid_df.loc[valid_mask, 'H'] / liquid_df.loc[valid_mask, 'S']
                max_temp = round(np.max(temps) + 200)
                min_temp = round(np.min(temps))
            else:
                max_temp = 2000
                min_temp = 300
        else:
            max_temp = 2000
            min_temp = 300
        
        # Apply physical temperature bounds
        # Don't go below ~100K (very low for any realistic metallurgical system)
        T_FLOOR = 100  # Kelvin - physical minimum temperature
        T_CEILING = 3500  # Kelvin - physical maximum temperature
        
        min_temp = max(min_temp - 200, T_FLOOR)
        max_temp = min(max_temp, T_CEILING)
        
        self.conds = [
            min_temp - self.temp_slider[0],
            max_temp + self.temp_slider[1]
        ]
        self.T_grid = np.arange(self.conds[0], self.conds[1] + self.T_incr, self.T_incr)
        
        # Create T-indexed groups
        self.df_Tgroups = {}
        for T in self.T_grid:
            temp_df = self._slice_hsx.copy()
            temp_df['G'] = temp_df['H'] - T * temp_df['S']
            self.df_Tgroups[T] = temp_df[['x0', 'x1', 'G', 'Phase', 'Colors']].copy()
        
        print(f"Temperature range: {self.conds[0]:.0f}K to {self.conds[1]:.0f}K")
        print(f"T grid points: {len(self.T_grid)}")
    
    def process_data(self) -> 'GeneralPlotter':
        """
        Process HSX data using T-loop approach (following ternary_HSX.py).
        
        For each temperature T:
        1. Compute G = H - T*S for all phases
        2. Compute 2D convex hull in (x0, x1, G) space
        3. Extract simplices with phase labels
        
        Returns:
            self (for method chaining)
        """
        self._init_sys()
        
        start_time = time.time()
        self.equil_df_list = []
        shifter = 0
        
        for T in self.T_grid:
            if T < self.conds[0]:
                continue
            
            points = np.array(self.df_Tgroups[T][['x0', 'x1', 'G']])
            simplices = gliq_lowerhull3(points, vertical_simplices=True)
            
            # Get phase labels for each simplex
            final_phases = []
            for simplex in simplices:
                phase1 = self.df_Tgroups[T].iloc[simplex[0]]['Phase']
                phase2 = self.df_Tgroups[T].iloc[simplex[1]]['Phase']
                phase3 = self.df_Tgroups[T].iloc[simplex[2]]['Phase']
                final_phases.append([phase1, phase2, phase3])
            
            # Build data for this temperature
            data = []
            last_val = 0
            for i, simplex in enumerate(simplices):
                labels = final_phases[i]
                if len(set(labels)) == 0:
                    continue
                
                x0_coords = [points[vertex][0] for vertex in simplex]
                x1_coords = [points[vertex][1] for vertex in simplex]
                
                for j, (x0, x1) in enumerate(zip(x0_coords, x1_coords)):
                    label = labels[j]
                    color = self.color_map.get(label, 'gray')
                    data.append([x0, x1, T, label, color, shifter + i])
                
                last_val = i
            
            shifter += (last_val + 1)
            
            if data:
                temp_df = pd.DataFrame(data, columns=['x0', 'x1', 'T', 'Phase', 'Colors', 'simplex_id'])
                
                # Store original coordinates before transformation
                temp_df['x0_orig'] = temp_df['x0'].copy()
                temp_df['x1_orig'] = temp_df['x1'].copy()
                
                # Convert to cartesian for plotting
                temp_df = cartesian_to_ternary_df(temp_df)
                temp_df['T'] = temp_df['T'] - 273.15  # Convert to Celsius
                
                self.equil_df_list.append(temp_df)
        
        elapsed = time.time() - start_time
        print(f"Lower hull evaluation time: {elapsed:.2f}s for T_incr={self.T_incr}K")
        
        return self
    
    # =========================================================================
    # Plotting (following ternary_HSX.py)
    # =========================================================================
    
    def _add_isothermal_lines(self, fig, liq_points, triangles):
        """Add isothermal contour lines to the 3D liquidus surface."""
        if len(liq_points) < 3:
            return
        
        temps = liq_points[:, 2]
        t_min, t_max = temps.min(), temps.max()
        t_range = t_max - t_min
        
        if t_range < 10:
            return
        
        # Choose appropriate delta_T
        if t_range <= 50:
            delta_T = 10
        elif t_range <= 100:
            delta_T = 20
        elif t_range <= 500:
            delta_T = 50
        else:
            delta_T = 100
        
        iso_temps = np.arange(np.ceil(t_min / delta_T) * delta_T, t_max, delta_T)
        
        # Collect all line segments
        all_x, all_y, all_z = [], [], []
        
        for iso_temp in iso_temps:
            for tri in triangles:
                v1, v2, v3 = liq_points[tri[0]], liq_points[tri[1]], liq_points[tri[2]]
                
                intersections = []
                edges = [(v1, v2), (v2, v3), (v3, v1)]
                
                for p1, p2 in edges:
                    t1, t2 = p1[2], p2[2]
                    if (t1 <= iso_temp <= t2) or (t2 <= iso_temp <= t1):
                        if abs(t2 - t1) > 1e-8:
                            alpha = (iso_temp - t1) / (t2 - t1)
                            if 0 <= alpha <= 1:
                                intersection = p1 + alpha * (p2 - p1)
                                intersection[2] = iso_temp
                                intersections.append(intersection)
                
                if len(intersections) == 2:
                    all_x.extend([intersections[0][0], intersections[1][0], None])
                    all_y.extend([intersections[0][1], intersections[1][1], None])
                    all_z.extend([intersections[0][2], intersections[1][2], None])
        
        if all_x:
            fig.add_trace(go.Scatter3d(
                x=all_x, y=all_y, z=all_z,
                mode='lines',
                line=dict(color='white', width=2),
                name='Isotherms',
                showlegend=False,
                hoverinfo='skip',
            ))
    
    def plot_ternary(self) -> go.Figure:
        """
        Generate 3D liquidus surface plot (following ternary_HSX.py).
        
        Returns:
            Plotly figure object.
        """
        if not self.equil_df_list:
            raise RuntimeError("Call process_data() first")
        
        fig = go.Figure()
        
        self.plotting_df = pd.concat(self.equil_df_list)
        simplex_df = deepcopy(self.plotting_df)
        
        # Process liquid and solid points
        liq_simplex_df = simplex_df[simplex_df['Phase'] == 'L']
        solid_simplex_df = simplex_df[simplex_df['Phase'] != 'L']
        liq_simplex_df = liq_simplex_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='first')
        simplex_df = pd.concat([solid_simplex_df, liq_simplex_df])
        
        # Filter valid simplices (exactly 3 vertices)
        id_counts = simplex_df["simplex_id"].value_counts()
        valid_ids = id_counts[id_counts == 3].index
        simplex_df = simplex_df[simplex_df['simplex_id'].isin(valid_ids)].copy()
        simplex_df = simplex_df.sort_values(by='simplex_id').reset_index(drop=True)
        
        self.liq_plotting_df = self.plotting_df[self.plotting_df['Phase'] == 'L']
        self.solid_plotting_df = self.plotting_df[self.plotting_df['Phase'] != 'L']
        self.solid_plotting_df = self.solid_plotting_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='last')
        
        # =====================================================================
        # FILTER LIQUID POINTS FIRST - Remove vertical simplex artifacts
        # =====================================================================
        self.liq_plotting_df = self.liq_plotting_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='first')
        
        n_before = len(self.liq_plotting_df)
        
        # Filter by temperature bounds (catches near-vertical simplex artifacts)
        self.liq_plotting_df = self.liq_plotting_df[
            (self.liq_plotting_df['T'] >= self.T_min_filter) & 
            (self.liq_plotting_df['T'] <= self.T_max_filter)
        ].copy()
        
        n_after = len(self.liq_plotting_df)
        if n_before != n_after:
            print(f"  Filtered {n_before - n_after} liquid points with extreme temperatures")
            print(f"    (T_min={self.T_min_filter}°C, T_max={self.T_max_filter}°C)")
        
        # Determine actual T range from filtered liquid data
        actual_t_min = self.liq_plotting_df['T'].min() if len(self.liq_plotting_df) > 0 else 0
        actual_t_max = self.liq_plotting_df['T'].max() if len(self.liq_plotting_df) > 0 else 1500
        t_base_solid = actual_t_min - 100  # Solid lines extend below min liquidus
        
        # Extend solid lines to base (using actual filtered T range)
        solid_base_rows = []
        for index, row in self.solid_plotting_df.iterrows():
            x0 = row['x0']
            x1 = row['x1']
            label = row['Phase']
            color = row['Colors']
            new_row = {'x0': x0, 'x1': x1, 'T': t_base_solid, 'Phase': label, 'Colors': color,
                      'x0_orig': row.get('x0_orig', x0), 'x1_orig': row.get('x1_orig', x1),
                      'simplex_id': -1}
            solid_base_rows.append(new_row)
        
        if solid_base_rows:
            self.solid_plotting_df = pd.concat([self.solid_plotting_df, pd.DataFrame(solid_base_rows)], ignore_index=True)
        
        # Build triangulation for liquidus surface
        liq_points = np.array(list(zip(
            self.liq_plotting_df['x0'],
            self.liq_plotting_df['x1'],
            self.liq_plotting_df['T']
        )))
        
        if len(liq_points) >= 3:
            cart_liq_points = [(p[0], p[1]) for p in liq_points]
            try:
                self.triangulation = Delaunay(cart_liq_points)
                triangles = self.triangulation.simplices
                
            except Exception as e:
                print(f"Warning: Delaunay triangulation failed: {e}")
                triangles = np.array([])
        else:
            triangles = np.array([])
        
        # Plot solid phase lines
        for label, group in self.solid_plotting_df.groupby('Phase'):
            fig.add_trace(go.Scatter3d(
                x=group['x0'], y=group['x1'], z=group['T'],
                mode='lines',
                line=dict(color=group['Colors'].iloc[0], width=10),
                showlegend=False,
                opacity=1,
                hovertemplate=f'<b>Phase: {label}</b><br><extra></extra>'
            ))
        
        # Plot liquidus mesh
        if len(triangles) > 0:
            fig.add_trace(go.Mesh3d(
                x=self.liq_plotting_df['x0'],
                y=self.liq_plotting_df['x1'],
                z=self.liq_plotting_df['T'],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.6,
                colorscale='Viridis',
                intensity=self.liq_plotting_df['T'],
                showscale=False,
                hovertemplate='<b>Liquidus Surface</b><br>' +
                            f'x_{self._slice_elements[1]}: %{{customdata[0]:.3f}}<br>' +
                            f'x_{self._slice_elements[2]}: %{{customdata[1]:.3f}}<br>' +
                            'T: %{z:.1f}°C<br><extra></extra>',
                customdata=np.column_stack((
                    self.liq_plotting_df['x0_orig'],
                    self.liq_plotting_df['x1_orig']
                ))
            ))
            
            # Add isothermal lines
            self._add_isothermal_lines(fig, liq_points, triangles)
        
        # Add legend entries for phases
        for phase, color in self.color_map.items():
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(color=color, size=10, opacity=1.0),
                name=phase,
                showlegend=True
            ))
        
        # Add ternary axes
        # Use actual filtered data range for base temperature
        actual_t_min = self.liq_plotting_df['T'].min() if len(self.liq_plotting_df) > 0 else self.conds[0] - 273.15
        actual_t_max = self.liq_plotting_df['T'].max() if len(self.liq_plotting_df) > 0 else self.conds[1] - 273.15
        t_base = actual_t_min - 50  # Base slightly below minimum liquidus temp
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3)/2, 0, 0],
            z=[t_base, t_base, t_base, t_base],
            mode='lines',
            line=dict(color='black', width=5),
            showlegend=False
        ))
        
        # Layout - use actual data range
        t_min_plot = t_base - 50
        t_max_plot = actual_t_max + 100
        
        fig.update_layout(
            title=f"Ternary Slice: {'-'.join(self._slice_elements)} at {self._fixed_element}={self._fixed_value*100:.0f}%",
            legend=dict(x=0.95, y=0.95, xanchor='left', yanchor='top'),
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50),
            scene=dict(
                zaxis=dict(range=[t_min_plot, t_max_plot], title='Temperature (°C)'),
                xaxis=dict(title=' ', showticklabels=False, showgrid=False),
                yaxis=dict(title=' ', showticklabels=False, showgrid=False),
            )
        )
        
        return fig
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_plotting_data(self) -> pd.DataFrame:
        """Get the processed plotting data."""
        if self.plotting_df is None:
            raise RuntimeError("Call process_data() first")
        return self.plotting_df.copy()
    
    def summary(self) -> Dict[str, any]:
        """Get summary of current state."""
        return {
            'n_components': self.n_components,
            'elements': self.elements,
            'n_hsx_points': len(self.hsx_data),
            'slice_defined': self._slice_defined,
            'fixed_element': self._fixed_element,
            'fixed_value': self._fixed_value,
            'slice_elements': self._slice_elements,
            'n_slice_points': len(self._slice_hsx) if self._slice_hsx is not None else 0,
            'T_range': self.conds if self.conds else None,
            'n_equil_dfs': len(self.equil_df_list),
            'T_min_filter': self.T_min_filter,
            'T_max_filter': self.T_max_filter,
            'tri_T_span_max': self.tri_T_span_max,
        }


if __name__ == "__main__":
    import plotly.offline as ploff
    from gliquid.config import data_dir
    
    # ===========================================================================
    # QUATERNARY SYSTEM: Ba-Mg-Si-Al
    # ===========================================================================
    
    print("="*70)
    print("QUATERNARY SYSTEM DEMONSTRATION: Ba-Mg-Si-Al")
    print("="*70)
    
    # System definition (alphabetically sorted for consistency)
    quat_sys = ["Al", "Ba", "Mg", "Si"]
    sorted_sys = sorted(quat_sys)  # ['Al', 'Ba', 'Mg', 'Si']
    
    # Generate all binary pairs for quaternary system (6 pairs)
    binary_sys_labels = []
    for i in range(len(sorted_sys)):
        for j in range(i + 1, len(sorted_sys)):
            binary_sys_labels.append(f"{sorted_sys[i]}-{sorted_sys[j]}")
    
    print(f"\nBinary pairs needed: {binary_sys_labels}")
    
    # Load binary parameters from fitted data
    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.25-filtered.xlsx")
    
    # Build binary L parameter dictionary with proper sign handling
    binary_L_dict = {}
    
    for bin_sys in binary_sys_labels:
        # Check if elements need to be flipped (sorted vs stored order)
        flipped_sys = "-".join(sorted(bin_sys.split('-')))
        order_changed = (bin_sys != flipped_sys)
        
        # Search for system in parameter DataFrame
        if bin_sys in binary_param_df['system'].tolist():
            params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
        elif flipped_sys in binary_param_df['system'].tolist():
            params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
        else:
            raise ValueError(f"Binary system {bin_sys} not found in parameter database")
        
        # Extract Redlich-Kister parameters
        L0_a = float(params["L0_a"])
        L0_b = float(params["L0_b"])
        L1_a = float(params["L1_a"])
        L1_b = float(params["L1_b"])
        
        # Flip L1 signs if element order was reversed
        # (L1 term is asymmetric: L1 * (x_A - x_B), sign depends on element order)
        if order_changed:
            L1_a = -L1_a
            L1_b = -L1_b
        
        binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]
        print(f"  {bin_sys}: L0=[{L0_a:.1f}, {L0_b:.4f}], L1=[{L1_a:.1f}, {L1_b:.4f}]")
    
    # ===========================================================================
    # STEP 1: Initialize GeneralInterpolation
    # ===========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: GeneralInterpolation Setup")
    print("="*70)
    
    output_dir = "all_dumps/quaternary_demo/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    interp = GeneralInterpolation(
        elements=sorted_sys,
        output_dir=output_dir,
        grid_delta=0.025,  # 5% composition steps (coarser for quaternary)
        param_format='combined',
        interp_scheme='linear'
    )
    
    # Set binary parameters
    interp.set_binary_params(binary_L_dict)
    
    print("\nSystem summary:")
    print(f"  Elements: {interp.elements}")
    print(f"  Binary pairs: {interp.binary_pairs}")
    print(f"  Grid delta: {interp.grid_delta}")
    
    # ===========================================================================
    # STEP 2: Run Interpolation (generates HSX data)
    # ===========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: Running Interpolation (this may take a while for MP fetch)")
    print("="*70)
    
    interp.interpolate(use_mp_cache=True)
    
    # Save HSX data
    hsx_csv_path = interp.save_hsx("quaternary_hsx_data.csv")
    print(f"\nHSX data saved to: {hsx_csv_path}")
    
    # ===========================================================================
    # STEP 3: Initialize GeneralPlotter and plot ternary slice
    # Using T-loop approach following ternary_HSX.py
    # ===========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: Ternary Slice Ba-Mg-Si at Al = 0%")
    print("="*70)
    
    # Create plotter for the ternary slice
    plotter = GeneralPlotter(
        hsx_data=interp.hsx_data,
        elements=sorted_sys,
        output_dir=output_dir,
        T_incr=10.0,  # 10K temperature increments (matching ternary_HSX.py)
    )
    
    # Define ternary slice by fixing Al = 0
    plotter.define_ternary_slice(
        fixed_element='Al',
        fixed_value=0.0,
        tolerance=0.01
    )
    
    # Process data using T-loop convex hull approach
    plotter.process_data()
    
    # Generate plot
    fig1 = plotter.plot_ternary()
    
    slice1_path = os.path.join(output_dir, "ternary_BaMgSi_Al0.html")
    ploff.plot(fig1, filename=slice1_path, auto_open=False)
    print(f"Saved: {slice1_path}")
    
    print(f"\nPlotter summary: {plotter.summary()}")
    
    # ===========================================================================
    # Export plotting data for verification
    # ===========================================================================
    
    print("\n" + "="*70)
    print("EXPORTING PLOTTING DATA")
    print("="*70)
    
    plotting_data = plotter.get_plotting_data()
    data_path = os.path.join(output_dir, "ternary_slice_data.csv")
    plotting_data.to_csv(data_path, index=False)
    print(f"Plotting data saved to: {data_path}")
    print(f"  Points: {len(plotting_data)}")
    print(f"  Temperature range: {plotting_data['T'].min():.1f}°C to {plotting_data['T'].max():.1f}°C")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"  - quaternary_hsx_data.csv")
    print(f"  - ternary_BaMgSi_Al0.html")
    print(f"  - ternary_slice_data.csv")