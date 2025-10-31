import numpy as np

# --- 1. Create a square matrix ---
A = np.array([
    [2, 3, 1],
    [4, 1, -3],
    [3, 2, 0]
], dtype=float)

print("Matrix A:")
print(A)

# --- 2. Determinant ---
det_A = np.linalg.det(A)
print("\nDeterminant of A:", round(det_A, 2))

# --- 3. Cofactor Matrix ---
def cofactor_matrix(M):
    n = M.shape[0]
    cof = np.zeros_like(M)
    for i in range(n):
        for j in range(n):
            # Minor of M[i, j]
            minor = np.delete(np.delete(M, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return cof

cofactor_A = cofactor_matrix(A)
print("\nCofactor Matrix of A:")
print(np.round(cofactor_A, 2))

# --- 4. Adjoint (Transpose of Cofactor Matrix) ---
adj_A = cofactor_A.T
print("\nAdjoint (Adjugate) of A:")
print(np.round(adj_A, 2))

# --- 5. Inverse (if determinant ≠ 0) ---
if det_A != 0:
    inv_A = adj_A / det_A
