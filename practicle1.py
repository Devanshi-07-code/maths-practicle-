import numpy as np

# Vector (simple)
v = np.array([1, 2, 3])
print("Vector:", v)

# Transpose of vector (2D banake)
vT = v.reshape(1, 3).T
print("Vector Transpose:\n", vT)

# Matrix
A = np.array([[1, 2],
              [3, 4]])
print("Matrix A:\n", A)

# Matrix transpose
print("Transpose of A:\n", A.T)

# Complex vector
z = np.array([1+2j, 4-3j])
print("Complex Vector:", z)

# Conjugate
print("Conjugate of z:", np.conj(z))

# Complex matrix
B = np.array([[1+1j, 2-2j],
              [3+0j, 4-1j]])

print("Matrix B:\n", B)

# Conjugate transpose of B
print("Conjugate Transpose of B:\n", B.conj().T)
