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




if __name__ == "__main__":
    # ===========================================================================
    # Example: ThermoExprBuilder (low-level expression building)
    # ===========================================================================
    print("=" * 70)
    print("EXAMPLE 1: ThermoExprBuilder (symbolic expression building)")
    print("=" * 70)
    
    # Build expressions for a ternary system
    builder = ThermoExprBuilder(
        n_components=3,
        param_format='combined',
        interp_scheme='linear'
    )
    
    # Load fusion data for demonstration
    with open(fusion_enthalpies_file) as f:
        fusion_enthalpies_data = json.load(f)
    with open(fusion_temps_file) as f:
        fusion_temps_data = json.load(f)
    
    elements = ['Al', 'Cu', 'Mg']
    fusion_H = [fusion_enthalpies_data[el] for el in sorted(elements)]
    fusion_T = [fusion_temps_data[el] for el in sorted(elements)]
    
    builder.set_reference_gibbs_from_fusion(fusion_H, fusion_T)
    builder.set_binary_L_params({
        (0, 1): [-5000, 0, 1000, 0],  # Al-Cu
        (0, 2): [-3000, 0, 500, 0],   # Al-Mg
        (1, 2): [-2000, 0, 300, 0],   # Cu-Mg
    })
    builder.build()
    
    print(f"\nSystem: {'-'.join(sorted(elements))}")
    print(f"Binary pairs: {builder.binary_pairs}")
    print(f"Mole fractions: {builder.mole_fractions}")
    print(f"\nExpressions built successfully!")
    
    # ===========================================================================
    # Example: GeneralInterpolation (full HSX pipeline)
    # ===========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: GeneralInterpolation (full HSX data generation)")
    print("=" * 70)
    
    # Initialize for a ternary system
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    interp = GeneralInterpolation(
        elements=['Al', 'Cu', 'Mg'],
        output_dir=output_dir,
        grid_delta=0.05,  # Coarser grid for quick demo
        param_format='combined',
        interp_scheme='linear'
    )
    
    # Set binary interaction parameters (example values)
    interp.set_binary_params({
        'Al-Cu': [-5000, 0, 1000, 0],
        'Al-Mg': [-3000, 0, 500, 0],
        'Cu-Mg': [-2000, 0, 300, 0],
    })
    
    # Run the full interpolation pipeline
    interp.interpolate(use_mp_cache=True)
    
    # Display summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    summary = interp.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Show a sample of the HSX data
    print("\n" + "-" * 40)
    print("HSX Data Sample (first 10 rows)")
    print("-" * 40)
    print(interp.hsx_data.head(10).to_string())