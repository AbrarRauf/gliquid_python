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
from extensive_hull_main import gliq_lowerhull3
import random


mpr = MPRester("Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ")  # Use environment variable for MP_API_KEY

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
    
    def build(self) -> 'ThermodynamicExpressionBuilder':
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
    




if __name__ == "__main__":
    # Example usage: build expressions for a quaternary system using the class-based API
    with open(fusion_enthalpies_file) as f:
        fusion_enthalpies_data = json.load(f)
    with open(fusion_temps_file) as f:
        fusion_temps_data = json.load(f)
    
    # Select specific elements for the quaternary system
    elements = ['Al', 'Cu', 'Si', 'Mg']
    fusion_H = [fusion_enthalpies_data[el] for el in elements]
    fusion_T = [fusion_temps_data[el] for el in elements]
    
    # Example binary parameters (these would come from fitting or literature)
    binary_params = {
        (0, 1): [1000, 0, 500, 0],  # L0_a, L0_b, L1_a, L1_b for pair (0,1)
        (0, 2): [1500, 0, 700, 0],
        (0, 3): [1300, 0, 650, 0],
        (1, 2): [1200, 0, 600, 0],
        (1, 3): [1100, 0, 550, 0],
        (2, 3): [1400, 0, 700, 0],
    }
    
    # =========================================================================
    # NEW CLASS-BASED API (recommended)
    # =========================================================================
    builder = ThermoExprBuilder(
        n_components=4,
        param_format='combined',
        interp_scheme='linear'
    )
    
    # Set inputs using class methods
    builder.set_reference_gibbs_from_fusion(fusion_H, fusion_T)
    builder.set_binary_L_params(binary_params)
    
    # Build expressions (returns self for chaining)
    builder.build()
    
    # Access expressions directly as attributes
    print("=" * 60)
    print("THERMODYNAMIC EXPRESSION BUILDER EXAMPLE")
    print("=" * 60)
    print(f"\nNumber of components: {builder.n_components}")
    print(f"Binary pairs: {builder.binary_pairs}")
    print(f"Independent mole fractions: {builder.x}")
    print(f"All mole fractions: {builder.mole_fractions}")
    print(f"\nG_liquid expression built successfully")
    print(f"S_liquid expression built successfully")
    print(f"H_liquid expression built successfully")
    
    print("\nG_liquid expression:")
    print(builder.g_liquid)
    print("\nS_liquid expression:")
    print(builder.s_liquid)
    print("\nH_liquid expression:")
    print(builder.h_liquid)

    # Example: lambdify and evaluate
    g_func = builder.lambdify('g_liquid')