import pandas as pd 
import numpy as np

def ternary_to_cartesian(x0, y0):
    # The same unit vectors used in your forward transform
    unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    
    # Invert the transformation matrix
    inv_unitvec = np.linalg.inv(unitvec)

    # Apply inverse transform
    original_coord = np.dot([x0, y0], inv_unitvec)
    return original_coord[0], original_coord[1]

# Example usage:
x0, y0 = 0.425, 0.234  # some ternary-transformed coordinate
x_orig, y_orig = ternary_to_cartesian(x0, y0)
print(f"Original Cartesian: ({x_orig:.5f}, {y_orig:.5f})")

