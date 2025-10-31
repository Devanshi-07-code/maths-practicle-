import numpy as np

# --- 1. Create a matrix ---
A = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [1, 1, 1]
], dtype=float)

print("Original Matrix A:")
print(A)

# --- 2. Function to convert to Row Echelon Form ---
def echelon_form(matrix):
    A = matrix.copy().astype(float)
    rows, cols = A.shape
    r = 0  # row index
    for c in range(cols):
        if r >= rows:
            break
        # Find the pivot (non-zero entry)
        pivot = np.argmax(np.abs(A[r:, c])) + r
        if A[pivot, c] == 0:
            continue
        # Swap current row with pivot row
        A[[r, pivot]] = A[[pivot, r]]
        # Normalize pivot row
        A[r] = A[r] / A[r, c]
        # Eliminate below pivot
        for i in range(r + 1, rows):
            A[i] = A[i] - A[i, c] * A[r]
        r += 1
    return A



