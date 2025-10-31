import numpy as np

# Create a vector and a matrix
v = np.array([1, 2, 3])                # Row vector
A = np.array([[1, 2, 3], [4, 5, 6]])   # Matrix

print("Vector v:\n", v)
print("\nMatrix A:\n", A)

# Transpose
print("\nTranspose of vector v:\n", v.T)   # Same as v (since it's 1D)
print("\nTranspose of matrix A:\n", A.T)

# Complex vector for conjugate transpose
v_complex = np.array([1 + 2j, 3 - 4j])
print("\nComplex vector v_complex:\n", v_complex)

# Conjugate transpose (Hermitian transpose)
v_H = np.conjugate(v_complex.T)
print("\nConjugate transpose of v_complex:\n", v_H)
