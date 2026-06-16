"""
Soal 11.1 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Selesaikan sistem tridiagonal berikut menggunakan Thomas Algorithm,
seperti pada (a) Contoh 11.1 dan (b) Contoh 11.3:

    [ 0.8  -0.4   0  ] [x1]   [ 41  ]
    [-0.4   0.8  -0.4] [x2] = [ 25  ]
    [ 0    -0.4   0.8] [x3]   [105  ]

Thomas Algorithm:
  1. Forward elimination (decomposition)
  2. Forward substitution
  3. Back substitution
"""

import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Selesaikan sistem tridiagonal [a, b, c] x = d menggunakan Thomas Algorithm.

    Parameter:
        a : list/array, diagonal bawah (a[0] tidak dipakai)
        b : list/array, diagonal utama
        c : list/array, diagonal atas (c[-1] tidak dipakai)
        d : list/array, vektor RHS

    Return:
        x : array solusi
    """
    n = len(d)
    # Salin agar tidak mengubah data asli
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # === Forward Sweep (Decomposition + Forward Sub) ===
    for i in range(1, n):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    # === Back Substitution ===
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def main():
    print("=" * 55)
    print("SOAL 11.1 – Thomas Algorithm untuk Sistem Tridiagonal")
    print("=" * 55)

    # Matriks tridiagonal:
    #   b (diagonal utama):  [0.8,  0.8,  0.8]
    #   a (diagonal bawah):  [0,   -0.4, -0.4]   (a[0] dummy)
    #   c (diagonal atas):   [-0.4, -0.4,  0]     (c[-1] dummy)
    #   d (RHS):             [41,   25,  105]

    b = [0.8,  0.8,  0.8]   # diagonal utama
    a = [0.0, -0.4, -0.4]   # diagonal bawah (a[0] tidak digunakan)
    c = [-0.4, -0.4, 0.0]   # diagonal atas  (c[-1] tidak digunakan)
    d = [41.0, 25.0, 105.0] # RHS

    print("\nSistem persamaan:")
    print("  [ 0.8  -0.4   0  ] [x1]   [ 41  ]")
    print("  [-0.4   0.8  -0.4] [x2] = [ 25  ]")
    print("  [ 0    -0.4   0.8] [x3]   [105  ]")

    x = thomas_algorithm(a, b, c, d)

    print("\n--- Solusi (Thomas Algorithm) ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.6f}")

    # Verifikasi dengan numpy.linalg.solve
    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])
    rhs = np.array([41.0, 25.0, 105.0])
    x_ref = np.linalg.solve(A, rhs)

    print("\n--- Verifikasi (numpy.linalg.solve) ---")
    for i, xi in enumerate(x_ref, 1):
        print(f"  x{i} = {xi:.6f}")

    resid = np.linalg.norm(A @ x - rhs)
    print(f"\nNorma residu ||Ax - b|| = {resid:.2e}")
    print("✓ Solusi valid!" if resid < 1e-10 else "✗ Periksa kembali!")


if __name__ == "__main__":
    main()
