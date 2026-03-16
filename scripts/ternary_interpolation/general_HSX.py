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


class SymbolicVariables:
    """
    Clean management of symbolic variables for multi-component thermodynamics.
    
    Uses indexed symbols for consistent naming across any number of components.
    All mole fractions are stored as x[0], x[1], ..., x[n-2] where n is the 
    number of components. The last component's fraction is computed as 
    1 - sum(x[i]).
    
    Attributes:
        n_components: Number of components in the system.
        t: Temperature symbol.
        x: Indexed mole fraction symbols (n-1 independent fractions).
    """
    
    def __init__(self, n_components: int):
        """
        Initialize symbolic variables for an n-component system.
        
        Args:
            n_components: Number of components (must be >= 2).
        """
        if n_components < 2:
            raise ValueError("System must have at least 2 components.")
        
        self.n_components = n_components
        
        # Temperature symbol
        self.t = sp.Symbol('T', real=True, positive=True)
        
        # Mole fraction symbols: x[0], x[1], ..., x[n-2]
        # x[n-1] is computed as 1 - sum(x[i]) for closure
        self.x = sp.symbols(f'x0:{n_components - 1}', real=True, nonnegative=True)
        if n_components == 2:
            # For binary, x is a single symbol, wrap in tuple for consistency
            self.x = (self.x,) if not isinstance(self.x, tuple) else self.x
    
    def get_mole_fractions(self) -> Tuple[sp.Expr, ...]:
        """
        Get all mole fractions including the dependent one.
        
        Returns:
            Tuple of n mole fraction expressions, where the last is computed
            from the closure constraint (sum = 1).
        """
        x_indep = list(self.x)
        x_dep = 1 - sum(self.x)
        return tuple(x_indep + [x_dep])
    
    def get_mole_fraction(self, index: int) -> sp.Expr:
        """
        Get mole fraction for component at given index.
        
        Args:
            index: Component index (0 to n-1).
            
        Returns:
            Mole fraction expression for that component.
        """
        if index < 0 or index >= self.n_components:
            raise IndexError(f"Component index {index} out of range [0, {self.n_components - 1}]")
        
        if index < self.n_components - 1:
            return self.x[index]
        else:
            return 1 - sum(self.x)


# ==============================================================================
# PARAMETER EXPRESSION TEMPLATES
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


# Registry of available parameter expressions
PARAM_EXPR_REGISTRY: Dict[str, Callable] = {
    'linear': _linear_expr,
    'exponential': _exponential_expr,
    'combined': _combined_expr,
}


def get_param_expr_func(param_format: str) -> Callable:
    """
    Get the parameter expression function for a given format.
    
    Args:
        param_format: One of 'linear', 'exponential', 'combined'.
        
    Returns:
        Function that takes (a, b, t) and returns sympy expression.
    """
    if param_format not in PARAM_EXPR_REGISTRY:
        raise ValueError(f"Unknown param_format '{param_format}'. "
                        f"Available: {list(PARAM_EXPR_REGISTRY.keys())}")
    return PARAM_EXPR_REGISTRY[param_format]


# ==============================================================================
# BINARY PAIR UTILITIES
# ==============================================================================

def get_ordered_binary_pairs(n_components: int) -> List[Tuple[int, int]]:
    """
    Generate all ordered binary pairs for n components.
    
    Binary pairs are ordered such that the first index < second index.
    For n=3: [(0,1), (0,2), (1,2)]
    For n=4: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    Args:
        n_components: Number of components.
        
    Returns:
        List of (i, j) tuples with i < j.
    """
    return list(combinations(range(n_components), 2))


def n_binary_pairs(n_components: int) -> int:
    """Number of binary pairs: n*(n-1)/2"""
    return n_components * (n_components - 1) // 2


# ==============================================================================
# INTERPOLATION SCHEMES
# ==============================================================================

def muggianu_weights(
    x: Tuple[sp.Expr, ...],
    i: int,
    j: int
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute Muggianu interpolation weights and effective binary compositions.
    
    The Muggianu model projects multicomponent compositions onto binary edges
    using symmetric weights. For components i and j:
        xi_eff = xi + (1 - xi - xj) / 2
        xj_eff = xj + (1 - xi - xj) / 2
        weight = xi * xj
    
    Args:
        x: Tuple of all mole fractions.
        i, j: Indices of the binary pair.
        
    Returns:
        (weight, xi_effective, xj_effective)
    """
    xi = x[i]
    xj = x[j]
    
    # Sum of other components
    remainder = 1 - xi - xj
    
    # Effective binary compositions (Muggianu projection)
    xi_eff = xi + remainder / 2
    xj_eff = xj + remainder / 2
    
    # Weight factor (simple xi*xj product)
    weight = xi * xj
    
    return weight, xi_eff, xj_eff


def kohler_weights(
    x: Tuple[sp.Expr, ...],
    i: int,
    j: int
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute Kohler interpolation weights and effective binary compositions.
    
    The Kohler model uses a different projection where binary compositions
    are normalized: xi_eff = xi / (xi + xj).
    
    Args:
        x: Tuple of all mole fractions.
        i, j: Indices of the binary pair.
        
    Returns:
        (weight, xi_effective, xj_effective)
    """
    xi = x[i]
    xj = x[j]
    
    # Effective binary compositions (Kohler projection)
    sum_ij = xi + xj
    # Use piecewise to handle division by zero
    xi_eff = sp.Piecewise((sp.Rational(1, 2), sp.Eq(sum_ij, 0)), (xi / sum_ij, True))
    xj_eff = sp.Piecewise((sp.Rational(1, 2), sp.Eq(sum_ij, 0)), (xj / sum_ij, True))
    
    # Weight factor
    weight = xi * xj
    
    return weight, xi_eff, xj_eff


def linear_interpolation_weights(
    x: Tuple[sp.Expr, ...],
    i: int,
    j: int
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Simple linear interpolation (equal weights = 1).
    
    This is the simplest model where binary parameters are used directly
    without composition-dependent weighting transformations.
    
    Args:
        x: Tuple of all mole fractions.
        i, j: Indices of the binary pair.
        
    Returns:
        (weight=1, xi, xj) - no transformation of compositions
    """
    return sp.Integer(1), x[i], x[j]


# Registry of interpolation schemes
INTERPOLATION_SCHEMES: Dict[str, Callable] = {
    'linear': linear_interpolation_weights,
    'muggianu': muggianu_weights,
    'kohler': kohler_weights,
}


# ==============================================================================
# CORE EXPRESSION BUILDERS
# ==============================================================================

def build_ideal_mixing_gibbs(
    x: Tuple[sp.Expr, ...],
    t: sp.Symbol,
    n_components: int
) -> sp.Expr:
    """
    Build the ideal mixing Gibbs energy expression.
    
    G_ideal = R*T * sum(xi * ln(xi)) for all components.
    
    Handles edge cases where some xi = 0 using piecewise logic.
    
    Args:
        x: Tuple of mole fraction expressions.
        t: Temperature symbol.
        n_components: Number of components.
        
    Returns:
        Sympy expression for ideal mixing Gibbs energy.
    """
    # Build sum of xi*ln(xi) terms with piecewise handling for xi=0
    mixing_terms = []
    for i in range(n_components):
        xi = x[i]
        # xi * ln(xi) -> 0 as xi -> 0
        term = sp.Piecewise(
            (0, sp.Eq(xi, 0)),
            (xi * sp.log(xi), True)
        )
        mixing_terms.append(term)
    
    return R * t * sum(mixing_terms)


def build_reference_gibbs(
    x: Tuple[sp.Expr, ...],
    g_ref_exprs: Tuple[sp.Expr, ...],
    n_components: int
) -> sp.Expr:
    """
    Build the reference Gibbs energy (mechanical mixture).
    
    G_ref = sum(xi * Gi_ref) for all components.
    
    Args:
        x: Tuple of mole fraction expressions.
        g_ref_exprs: Tuple of reference Gibbs energy expressions for each component.
        n_components: Number of components.
        
    Returns:
        Sympy expression for reference Gibbs energy.
    """
    if len(g_ref_exprs) != n_components:
        raise ValueError(f"Expected {n_components} reference energies, got {len(g_ref_exprs)}")
    
    return sum(x[i] * g_ref_exprs[i] for i in range(n_components))


def build_binary_excess_gibbs(
    x: Tuple[sp.Expr, ...],
    i: int,
    j: int,
    l0_expr: sp.Expr,
    l1_expr: sp.Expr,
    weight_func: Callable,
    weight_factor: sp.Expr = sp.Integer(1)
) -> sp.Expr:
    """
    Build excess Gibbs energy contribution from one binary pair.
    
    Uses 2-term Redlich-Kister expansion:
    G_xs_ij = weight * xi * xj * (L0 + L1 * (xi - xj))
    
    For ternary and higher systems, the weight_func transforms
    compositions according to the interpolation scheme.
    
    Args:
        x: Tuple of mole fraction expressions.
        i, j: Indices of the binary pair.
        l0_expr: L0 parameter expression (already with numeric values substituted).
        l1_expr: L1 parameter expression.
        weight_func: Function to compute interpolation weight.
        weight_factor: Additional weighting factor (default 1).
        
    Returns:
        Sympy expression for this binary's excess Gibbs contribution.
    """
    weight, xi_eff, xj_eff = weight_func(x, i, j)
    
    # Redlich-Kister 2-term polynomial
    rk_expr = l0_expr + l1_expr * (xi_eff - xj_eff)
    
    # Full contribution: weight * weight_factor * RK
    return weight * weight_factor * rk_expr


def build_thermodynamic_expressions(
    n_components: int,
    g_ref_exprs: Optional[Tuple[sp.Expr, ...]] = None,
    binary_L_exprs: Optional[Dict[Tuple[int, int], Tuple[sp.Expr, sp.Expr]]] = None,
    param_format: str = 'linear',
    interp_scheme: str = 'linear',
    binary_weights: Optional[Dict[Tuple[int, int], sp.Expr]] = None,
    **kwargs
) -> Dict[str, sp.Expr]:
    """
    Build thermodynamic expressions for an n-component system.
    
    This is the main function for constructing symbolic expressions for 
    Gibbs energy, enthalpy, and entropy of multi-component liquid phases.
    
    The excess Gibbs energy is built from binary interaction parameters
    using the specified interpolation scheme (Muggianu, Kohler, or linear).
    
    Args:
        n_components: Number of components in the system (>= 2).
        
        g_ref_exprs: Tuple of reference Gibbs energy expressions for each 
            component. If None, placeholder symbols are created.
            These should be functions of temperature (e.g., H - T*S).
            
        binary_L_exprs: Dictionary mapping binary pair indices (i, j) to 
            tuples of (L0_expr, L1_expr). Keys are tuples like (0, 1) for
            the binary between components 0 and 1.
            If None, placeholder symbols are created.
            
        param_format: Format for L parameter temperature dependence.
            Options: 'linear' (default), 'exponential', 'combined'.
            Only used when creating placeholder expressions.
            
        interp_scheme: Interpolation scheme for multi-component systems.
            Options: 'linear' (default), 'muggianu', 'kohler'.
            
        binary_weights: Optional dictionary of weight factors for each binary.
            Keys are (i, j) tuples. Default is 1 for all pairs.
            
        **kwargs: Additional optional arguments:
            - higher_order_exprs: Dict mapping tuple of component indices to
              expressions for higher-order (ternary, quaternary) interactions.
              E.g., {(0,1,2): L_ternary_expr} for ternary term.
            - tau: Time constant for combined expression (default 8000).
    
    Returns:
        Dictionary with keys:
            - 'symbols': SymbolicVariables instance
            - 'g_ref': Reference Gibbs energy expression
            - 'g_ideal': Ideal mixing Gibbs energy expression  
            - 'g_excess': Excess Gibbs energy expression
            - 'g_liquid': Total Gibbs energy (g_ref + g_ideal + g_excess)
            - 's_liquid': Entropy (-dG/dT)
            - 'h_liquid': Enthalpy (G + T*S)
            - 'mole_fractions': Tuple of all mole fraction expressions
            - 'binary_pairs': List of binary pair indices
    
    Example:
        >>> # Build expressions for a ternary system
        >>> exprs = build_thermodynamic_expressions(n_components=3)
        >>> G = exprs['g_liquid']
        >>> # Substitute specific parameter values
        >>> G_numeric = G.subs({...})
    """
    # Initialize symbolic variables
    symbols = SymbolicVariables(n_components)
    t = symbols.t
    x = symbols.get_mole_fractions()
    
    # Get interpolation scheme
    if interp_scheme not in INTERPOLATION_SCHEMES:
        raise ValueError(f"Unknown interp_scheme '{interp_scheme}'. "
                        f"Available: {list(INTERPOLATION_SCHEMES.keys())}")
    weight_func = INTERPOLATION_SCHEMES[interp_scheme]
    
    # Create reference Gibbs expressions if not provided
    if g_ref_exprs is None:
        g_ref_exprs = tuple(
            sp.Symbol(f'G_ref_{i}') for i in range(n_components)
        )
    
    # Get all binary pairs
    binary_pairs = get_ordered_binary_pairs(n_components)
    
    # Create binary L expressions if not provided  
    if binary_L_exprs is None:
        binary_L_exprs = {}
        for i, j in binary_pairs:
            l0_sym = sp.Symbol(f'L0_{i}{j}')
            l1_sym = sp.Symbol(f'L1_{i}{j}')
            binary_L_exprs[(i, j)] = (l0_sym, l1_sym)
    
    # Setup binary weights (default = 1)
    if binary_weights is None:
        binary_weights = {pair: sp.Integer(1) for pair in binary_pairs}
    
    # Build reference Gibbs energy
    g_ref = build_reference_gibbs(x, g_ref_exprs, n_components)
    
    # Build ideal mixing Gibbs energy
    g_ideal = build_ideal_mixing_gibbs(x, t, n_components)
    
    # Build excess Gibbs energy from all binaries
    g_excess_terms = []
    for pair in binary_pairs:
        i, j = pair
        l0, l1 = binary_L_exprs.get(pair, (sp.Integer(0), sp.Integer(0)))
        weight = binary_weights.get(pair, sp.Integer(1))
        
        g_xs_ij = build_binary_excess_gibbs(
            x, i, j, l0, l1, weight_func, weight
        )
        g_excess_terms.append(g_xs_ij)
    
    g_excess = sum(g_excess_terms)
    
    # Add higher-order terms if provided
    higher_order_exprs = kwargs.get('higher_order_exprs', {})
    for indices, expr in higher_order_exprs.items():
        # Higher order term: L * product(xi for i in indices)
        product_term = sp.Integer(1)
        for idx in indices:
            product_term *= x[idx]
        g_excess += expr * product_term
    
    # Total Gibbs energy
    g_liquid = g_ref + g_ideal + g_excess
    
    # Entropy: S = -dG/dT at constant P, composition
    s_liquid = -sp.diff(g_liquid, t)
    
    # Enthalpy: H = G + T*S
    h_liquid = g_liquid + t * s_liquid
    
    return {
        'symbols': symbols,
        'g_ref': g_ref,
        'g_ideal': g_ideal,
        'g_excess': g_excess,
        'g_liquid': g_liquid,
        's_liquid': s_liquid,
        'h_liquid': h_liquid,
        'mole_fractions': x,
        'binary_pairs': binary_pairs,
    }


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON USE CASES
# ==============================================================================

def create_reference_gibbs_from_fusion(
    fusion_enthalpies: List[float],
    fusion_temps: List[float],
    t: sp.Symbol
) -> Tuple[sp.Expr, ...]:
    """
    Create reference Gibbs energy expressions from fusion data.
    
    G_ref_i = H_fus_i - T * S_fus_i, where S_fus = H_fus / T_fus.
    
    Args:
        fusion_enthalpies: List of fusion enthalpies [J/mol] for each component.
        fusion_temps: List of fusion temperatures [K] for each component.
        t: Temperature symbol.
        
    Returns:
        Tuple of reference Gibbs energy expressions.
    """
    n = len(fusion_enthalpies)
    if len(fusion_temps) != n:
        raise ValueError("fusion_enthalpies and fusion_temps must have same length")
    
    g_refs = []
    for h_fus, t_fus in zip(fusion_enthalpies, fusion_temps):
        s_fus = h_fus / t_fus
        g_ref = h_fus - t * s_fus
        g_refs.append(g_ref)
    
    return tuple(g_refs)


def create_L_expressions_from_params(
    binary_params: Dict[Tuple[int, int], List[float]],
    param_format: str,
    t: sp.Symbol,
    **kwargs
) -> Dict[Tuple[int, int], Tuple[sp.Expr, sp.Expr]]:
    """
    Create L0/L1 expressions from numeric parameters.
    
    Args:
        binary_params: Dict mapping (i, j) to [L0_a, L0_b, L1_a, L1_b] parameters.
            For 'linear': L = a + b*T
            For 'exponential': L = a * exp(-T/b)
            For 'combined': L = (a + b*T) * exp(-T/tau)
            
        param_format: One of 'linear', 'exponential', 'combined'.
        t: Temperature symbol.
        **kwargs: Additional args like 'tau' for combined format.
        
    Returns:
        Dict mapping (i, j) to (L0_expr, L1_expr) symbolic expressions.
    """
    expr_func = get_param_expr_func(param_format)
    tau = kwargs.get('tau', sp.Integer(8000))
    
    result = {}
    for pair, params in binary_params.items():
        if param_format == 'linear':
            l0 = expr_func(params[0], params[1], t)
            l1 = expr_func(params[2], params[3], t)
        elif param_format == 'exponential':
            l0 = expr_func(params[0], params[1], t)
            l1 = expr_func(params[2], params[3], t)
        elif param_format == 'combined':
            l0 = _combined_expr(params[0], params[1], t, tau)
            l1 = _combined_expr(params[2], params[3], t, tau)
        else:
            raise ValueError(f"Unknown param_format: {param_format}")
        
        result[pair] = (l0, l1)
    
    return result



if __name__ == "__main__":
    # Example usage: build expressions for a quaternary system with fusion data
    with open(fusion_enthalpies_file) as f:
        fusion_enthalpies_data = json.load(f)
    with open(fusion_temps_file) as f:
        fusion_temps_data = json.load(f)
    
    # Select specific elements for the quaternary system
    elements = ['Al', 'Cu', 'Si', 'Mg']
    fusion_H = [fusion_enthalpies_data[el] for el in elements]
    fusion_T = [fusion_temps_data[el] for el in elements]
    
    g_ref_exprs = create_reference_gibbs_from_fusion(
        fusion_H, fusion_T, sp.Symbol('T')
    )
    
    # Example binary parameters (these would come from fitting or literature)
    binary_params = {
        (0, 1): [1000, 0, 500, 0],  # L0_a, L0_b, L1_a, L1_b for pair (0,1)
        (0, 2): [1500, 0, 700, 0],
        (0, 3): [1300, 0, 650, 0],
        (1, 2): [1200, 0, 600, 0],
        (1, 3): [1100, 0, 550, 0],
        (2, 3): [1400, 0, 700, 0],
    }
    
    binary_L_exprs = create_L_expressions_from_params(
        binary_params, 'linear', sp.Symbol('T')
    )
    
    expressions = build_thermodynamic_expressions(
        n_components=4,
        g_ref_exprs=g_ref_exprs,
        binary_L_exprs=binary_L_exprs,
        param_format='linear',
        interp_scheme='muggianu'
    )
    
    print("Gibbs energy expression for liquid phase:")
    print(expressions['g_liquid'])
    print(expressions['s_liquid'])
    print(expressions['h_liquid'])