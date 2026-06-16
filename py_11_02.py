"""
Soal 11.2 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Tentukan invers matriks untuk sistem Contoh 11.1 menggunakan
dekomposisi LU dan vektor unit.

Sistem (Contoh 11.1):
    [ 0.8  -0.4   0  ] [x1]   [ 41  ]
    [-0.4   0.8  -0.4] [x2] = [ 25  ]
    [ 0    -0.4   0.8] [x3]   [105  ]

Metode:
  1. Dekomposisi LU (tanpa pivoting, cocok untuk tridiagonal)
  2. Selesaikan [L][U][x] = [e_i] untuk setiap vektor unit e_i
  3. Kolom-kolom solusi membentuk invers matriks A^{-1}
"""

import numpy as np
from scipy.linalg import lu


def lu_decompose(A):
    """
    Dekomposisi LU tanpa pivoting (Doolittle).
    Return L, U sehingga A = L @ U.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)

    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ZeroDivisionError("Elemen pivot nol – gunakan pivoting!")
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U


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


def lu_inverse(A):
    """Hitung invers A menggunakan LU decomposition + vektor unit."""
    n = A.shape[0]
    L, U = lu_decompose(A)
    A_inv = np.zeros((n, n))

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0          # vektor unit ke-j
        y = forward_sub(L, e)
        x = back_sub(U, y)
        A_inv[:, j] = x     # kolom j dari A^{-1}

    return A_inv, L, U


def main():
    print("=" * 55)
    print("SOAL 11.2 – Invers Matriks via LU Decomposition")
    print("=" * 55)

    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])

    print("\nMatriks A:")
    print(A)

    A_inv, L, U = lu_inverse(A)

    print("\n--- Dekomposisi LU ---")
    print("Matriks L:")
    print(np.round(L, 6))
    print("\nMatriks U:")
    print(np.round(U, 6))

    print("\n--- Invers A (via LU + vektor unit) ---")
    print(np.round(A_inv, 6))

    # Verifikasi: A @ A_inv harus = I
    product = A @ A_inv
    print("\n--- Verifikasi: A × A⁻¹ ---")
    print(np.round(product, 6))

    identity_check = np.allclose(product, np.eye(3))
    print(f"\n{'✓ A × A⁻¹ = I (benar)' if identity_check else '✗ Gagal!'}")

    # Bandingkan dengan numpy
    A_inv_np = np.linalg.inv(A)
    print("\n--- Referensi: numpy.linalg.inv(A) ---")
    print(np.round(A_inv_np, 6))

    # Solusi sistem menggunakan invers
    b = np.array([41.0, 25.0, 105.0])
    x = A_inv @ b
    print("\n--- Solusi x = A⁻¹ b ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.6f}")


if __name__ == "__main__":
    main()
