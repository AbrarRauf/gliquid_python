import json
from general_HSX import ThermoExprBuilder
from gliquid.config import fusion_enthalpies_file, fusion_temps_file
from pathlib import Path

WORKSPACE_ROOT = Path("C:\\Users\\AbrarRauf\\University of Michigan Dropbox\\Abrar Rauf\\WHSun_Lab\\G_liquid")

# Example usage: build expressions for a quaternary system using the class-based API
with open(fusion_enthalpies_file) as f:
    fusion_enthalpies_data = json.load(f)
with open(fusion_temps_file) as f:
    fusion_temps_data = json.load(f)

# Select specific elements for the quaternary system
elements = ['Al', 'Cu', 'Si', 'Mg']
fusion_H = [fusion_enthalpies_data[el] for el in elements]
fusion_T = [fusion_temps_data[el] for el in elements]

# Dummy binary parameters
binary_params = {
    (0, 1): [1000, 0, 500, 0],  # L0_a, L0_b, L1_a, L1_b for pair (0,1)
    (0, 2): [1500, 0, 700, 0],
    (0, 3): [1300, 0, 650, 0],
    (1, 2): [1200, 0, 600, 0],
    (1, 3): [1100, 0, 550, 0],
    (2, 3): [1400, 0, 700, 0],
}

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
print(f"\nLambdified G_liquid function evaluated at x0=0.25, x1=0.25, x2=0.25, T=1000K:")
print(g_func(0.25, 0.25, 0.25, 1000))

