import numpy as np

# -----------------------------------------------------
# 1️⃣ Function: Check Linear Dependence
# -----------------------------------------------------
def check_linear_dependence(vectors):
    """
    Check if the given vectors are linearly dependent.
    """
    flat_vectors = [np.array(v).flatten() for v in vectors]
    M = np.column_stack(flat_vectors)
    rank = np.linalg.matrix_rank(M)
    dependent = rank < M.shape[1]
    return dependent, rank


# -----------------------------------------------------
# 2️⃣ Function: Generate Basis
# -----------------------------------------------------
def generate_basis(vectors, tol=1e-10):
    """
    Generate an independent basis from given vectors.
    """
    flat_vectors = [np.array(v).flatten() for v in vectors]
    M = np.column_stack(flat_vectors)
    U, S, Vt = np.linalg.svd(M)
    rank = np.sum(S > tol)
    basis = U[:, :rank]
    return basis


# -----------------------------------------------------
# 3️⃣ Function: Transition Matrix
# -----------------------------------------------------
def transition_matrix(basis_from, basis_to):
    """
    Find the transition matrix that converts coordinates
    from basis_from to basis_to.
    """
    B_from = np.column_stack([np.array(b).flatten() for b in basis_from])
    B_to = np.column_stack([np.array(b).flatten() for b in basis_to])
    P = np.linalg.solve(B_to, B_from)
    return P


# -----------------------------------------------------
# 🧮 MAIN PROGRAM (User-Defined Input)
# -----------------------------------------------------
print("=== Linear Algebra Operations ===")
print("1. Check Linear Dependence")
print("2. Generate Basis")
print("3. Find Transition Matrix")
print("---------------------------------\n")

choice = int(input("Enter your choice (1/2/3): "))

# --- 1️⃣ Check Linear Dependence ---
if choice == 1:
    n = int(input("Enter the number of vectors: "))
    vectors = []
    for i in range(n):
        v = list(map(float, input(f"Enter vector {i+1} elements (space-separated): ").split()))
        vectors.append(v)

    dep, rank = check_linear_dependence(vectors)
    print("\nLinear Dependence Check Result:")
    print("Dependent:", dep)
    print("Rank of the set:", rank)

# --- 2️⃣ Generate Basis ---
elif choice == 2:
    n = int(input("Enter the number of vectors: "))
    vectors = []
    for i in range(n):
        v = list(map(float, input(f"Enter vector {i+1} elements (space-separated): ").split()))
        vectors.append(v)

    basis = generate_basis(vectors)
    print("\nBasis for the Given Vectors:")
    print(np.round(basis, 3))
    print("Dimension of Basis (Rank):", basis.shape[1])

# --- 3️⃣ Transition Matrix ---
elif choice == 3:
    print("\nDefine FIRST basis (Basis A):")
    n = int(input("Enter number of basis vectors: "))
    basis_A = []
    for i in range(n):
        b = list(map(float, input(f"Enter basis A vector {i+1} (space-separated): ").split()))
        basis_A.append(b)

    print("\nDefine SECOND basis (Basis B):")
    basis_B = []
    for i in range(n):
        b = list(map(float, input(f"Enter basis B vector {i+1} (space-separated): ").split()))
        basis_B.append(b)

    P = transition_matrix(basis_A, basis_B)
    print("\nTransition Matrix (From Basis A → Basis B):")
    print(np.round(P, 3))

else:
    print("Invalid choice. Please restart the program.")

