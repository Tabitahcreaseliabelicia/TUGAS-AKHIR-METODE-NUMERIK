"""
Soal 11.3 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Sistem tridiagonal besar (Crank-Nicolson) diselesaikan menggunakan
Thomas Algorithm.

Sistem 4×4 (dari gambar buku):
    [ 2.01475  -0.020875   0          0        ] [T1]   [4.175  ]
    [-0.020875   2.01475  -0.020875   0        ] [T2] = [0      ]
    [ 0         -0.020875  2.01475   -0.020875 ] [T3]   [0      ]
    [ 0          0        -0.020875   2.01475  ] [T4]   [2.0875 ]

Gunakan Thomas Algorithm (sama seperti Soal 11.1).
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """
    Thomas Algorithm untuk sistem tridiagonal.
    a: diagonal bawah, b: diagonal utama, c: diagonal atas, d: RHS
    """
    n = len(d)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # Forward sweep
    for i in range(1, n):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def main():
    print("=" * 60)
    print("SOAL 11.3 – Thomas Algorithm untuk Sistem Tridiagonal Besar")
    print("       (Crank-Nicolson / Partial Differential Equations)")
    print("=" * 60)

    # Koefisien tridiagonal
    diag_main = 2.01475            # diagonal utama
    diag_off  = -0.020875          # off-diagonal (atas & bawah)
    n = 4

    b = [diag_main] * n
    a = [0.0] + [diag_off] * (n - 1)   # a[0] tidak dipakai
    c = [diag_off] * (n - 1) + [0.0]   # c[-1] tidak dipakai
    d = [4.175, 0.0, 0.0, 2.0875]

    print("\nSistem persamaan (4×4 tridiagonal):")
    print("  [ 2.01475  -0.020875   0          0        ] [T1]   [4.175  ]")
    print("  [-0.020875  2.01475  -0.020875    0        ] [T2] = [0      ]")
    print("  [ 0        -0.020875   2.01475   -0.020875 ] [T3]   [0      ]")
    print("  [ 0         0         -0.020875   2.01475  ] [T4]   [2.0875 ]")

    T = thomas_algorithm(a, b, c, d)

    print("\n--- Solusi (Thomas Algorithm) ---")
    for i, Ti in enumerate(T, 1):
        print(f"  T{i} = {Ti:.6f}")

    # Verifikasi
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diag_main
        if i > 0:
            A[i, i-1] = diag_off
        if i < n - 1:
            A[i, i+1] = diag_off

    rhs = np.array(d)
    resid = np.linalg.norm(A @ T - rhs)
    print(f"\nNorma residu ||AT - b|| = {resid:.2e}")

    # Referensi numpy
    T_ref = np.linalg.solve(A, rhs)
    print("\n--- Referensi (numpy.linalg.solve) ---")
    for i, Ti in enumerate(T_ref, 1):
        print(f"  T{i} = {Ti:.6f}")

    print("✓ Solusi valid!" if resid < 1e-10 else "✗ Periksa kembali!")


if __name__ == "__main__":
    main()
