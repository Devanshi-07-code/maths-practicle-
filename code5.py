import numpy as np

# === USER-DEFINED FUNCTION ===
def gaussian_elimination(A, b):
    """
    Solve a system of linear equations using Gaussian elimination.
    Handles both homogeneous (b = 0) and non-homogeneous systems.
    """
    A = A.astype(float)
    b = b.astype(float)

    # Create augmented matrix [A|b]
    Ab = np.hstack((A, b))
    n = len(b)

    # Forward elimination
    for i in range(n):
        # Pivoting (swap if diagonal is zero)
        if Ab[i, i] == 0:
            for j in range(i + 1, n):
                if Ab[j, i] != 0:
                    Ab[[i, j]] = Ab[[j, i]]
                    break

        # Make the pivot = 1
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminate below pivot
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])

    return x, Ab


# === MAIN PROGRAM ===

# Coefficient matrix (A)
A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
], dtype=float)

# Constant matrix (b)
b = np.array([[8], [-11], [-3]], dtype=float)

# Solve non-homogeneous system
x_non_homog, U_non_homog = gaussian_elimination(A, b)
print("Upper Triangular Form (Non-Homogeneous):")
print(np.round(U_non_homog, 2))
print("\nSolution for Non-Homogeneous System (Ax = b):")
print(np.round(x_non_homog, 2))

# Solve homogeneous system (b = 0)
b_zero = np.zeros((A.shape[0], 1))
x_homog, U_homog = gaussian_elimination(A, b_zero)
print("\nUpper Triangular Form (Homogeneous):")
print(np.round(U_homog, 2))
print("\nHomogeneous Solution:")
print(np.round(x_homog, 2))
