"""
Soal 11.5 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Lakukan kalkulasi yang sama seperti Contoh 11.2, tetapi untuk
sistem simetris berikut. Selain menyelesaikan dekomposisi Cholesky,
selesaikan juga nilai a's.

Sistem:
    [  6   15    55] [a0]   [152.6 ]
    [ 15   55   225] [a1] = [585.6 ]
    [ 55  225   979] [a2]   [2488.8]

Langkah:
  1. Cholesky decomposition: A = L @ L^T
  2. Forward substitution: L d = b
  3. Back substitution: L^T a = d
"""

import numpy as np


def cholesky_decompose(A):
    """Dekomposisi Cholesky: A = L @ L^T."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def forward_sub(L, b):
    """Substitusi maju untuk Ly = b."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_sub(U, y):
    """Substitusi mundur untuk Ux = y."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def main():
    print("=" * 60)
    print("SOAL 11.5 – Cholesky Decomposition + Penyelesaian Sistem")
    print("=" * 60)

    A = np.array([
        [  6,  15,   55],
        [ 15,  55,  225],
        [ 55, 225,  979]
    ], dtype=float)

    b = np.array([152.6, 585.6, 2488.8])

    print("\nMatriks A:")
    print(A)
    print("\nVektor b:")
    print(b)

    # Cholesky
    L = cholesky_decompose(A)
    print("\n--- Matriks L (Cholesky) ---")
    print(np.round(L, 6))

    # Forward substitution: L d = b
    d = forward_sub(L, b)
    print("\n--- Substitusi Maju: L·d = b ---")
    print(f"  d = {np.round(d, 6)}")

    # Back substitution: L^T a = d
    a = back_sub(L.T, d)
    print("\n--- Substitusi Mundur: Lᵀ·a = d ---")
    for i, ai in enumerate(a):
        print(f"  a{i} = {ai:.6f}")

    # Verifikasi
    resid = np.linalg.norm(A @ a - b)
    print(f"\nNorma residu ||Aa - b|| = {resid:.2e}")

    # Referensi numpy
    a_ref = np.linalg.solve(A, b)
    print("\n--- Referensi (numpy.linalg.solve) ---")
    for i, ai in enumerate(a_ref):
        print(f"  a{i} = {ai:.6f}")

    print("✓ Solusi valid!" if resid < 1e-8 else "✗ Periksa kembali!")


if __name__ == "__main__":
    main()
