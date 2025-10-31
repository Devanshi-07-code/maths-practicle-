import numpy as np
from scipy.linalg import null_space

# === USER-DEFINED FUNCTION ===
def find_subspaces(A):
    """Find basis for column space, row space, null space, and left null space."""
    
    # Convert to float array
    A = np.array(A, dtype=float)

    # Rank of matrix
    rank = np.linalg.matrix_rank(A)
    
    # --- Column Space (Range) ---
    Q, _ = np.linalg.qr(A)  # QR decomposition
    col_space = Q[:, :rank]

    # --- Row Space ---
    Qr, _ = np.linalg.qr(A.T)
    row_space = Qr[:, :rank]

    # --- Null Space ---
    null_sp = null_space(A)

    # --- Left Null Space ---
    left_null_sp = null_space(A.T)

    # Return results
    return col_space, row_space, null_sp, left_null_sp


# === MAIN PROGRAM ===

# Define the matrix
A = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [1, 1, 1]
])

print("Matrix A:")
print(A)

# Get all subspaces
col_space, row_space, null_sp, left_null_sp = find_subspaces(A)

# Display results
print("\nBasis for Column Space:")
print(np.round(col_space, 2))

print("\nBasis for Row Space:")
print(np.round(row_space, 2))

print("\nBasis for Null Space:")
print(np.round(null_sp, 2))

print("\nBasis for Left Null Space:")
print(np.round(left_null_sp, 2))
