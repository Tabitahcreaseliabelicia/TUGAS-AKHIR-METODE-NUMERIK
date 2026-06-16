"""
Soal 11.28 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Sistem PENTADIAGONAL memiliki bandwidth 5 (seperti bandwidth tridiagonal = 3).
Bentuk umum:

    [f1  g1  h1  0   0 ] [x1]   [r1]
    [e2  f2  g2  h2  0 ] [x2]   [r2]
    [d3  e3  f3  g3  h3] [x3] = [r3]
    [.                 ] [.]    [.]
    [dn  en  fn       ] [xn]   [rn]

Kembangkan program untuk menyelesaikan sistem pentadiagonal secara efisien
TANPA pivoting, mirip dengan Thomas Algorithm untuk tridiagonal.

Uji dengan sistem berikut (Contoh 11.1.1):
    [ 8  -2  -1   0   0] [x1]   [5]
    [-2   9  -4  -1   0] [x2]   [2]
    [-1  -3   7  -1  -2] [x3] = [0]
    [ 0  -4  -2  12  -5] [x4]   [1]
    [ 0   0  -7  -3  15] [x5]   [5]
"""

import numpy as np


def pentadiagonal_solver(d, e, f, g, h, r):
    """
    Selesaikan sistem pentadiagonal tanpa pivoting.

    Notasi band (5-diagonal):
        d : sub-sub-diagonal (d[2..n-1], d[0..1] tidak dipakai)
        e : sub-diagonal     (e[1..n-1], e[0] tidak dipakai)
        f : diagonal utama   (f[0..n-1])
        g : super-diagonal   (g[0..n-2], g[n-1] tidak dipakai)
        h : super-super-diag (h[0..n-3], h[n-2..n-1] tidak dipakai)
        r : vektor RHS

    Algoritma:
        Forward elimination mengurangi sistem ke tridiagonal kemudian diagonal,
        diikuti back substitution.
    """
    n = len(r)
    d = np.array(d, dtype=float)
    e = np.array(e, dtype=float)
    f = np.array(f, dtype=float)
    g = np.array(g, dtype=float)
    h = np.array(h, dtype=float)
    r = np.array(r, dtype=float)

    print(f"\n  === Forward Elimination (n={n}) ===")
    print(f"  {'Langkah':<10} {'Pivot':<10} {'Operasi'}")

    for i in range(n - 1):
        # Eliminasi baris i+1 menggunakan baris i
        if abs(f[i]) < 1e-14:
            raise ZeroDivisionError(f"Pivot nol di baris {i}! Gunakan pivoting.")

        # Eliminasi elemen sub-diagonal e[i+1]
        if i + 1 < n:
            factor1 = e[i + 1] / f[i]
            f[i + 1] -= factor1 * g[i]
            r[i + 1] -= factor1 * r[i]
            if i + 1 < n - 1:
                g[i + 1] -= factor1 * h[i]
            e[i + 1] = 0.0
            print(f"  Elim e[{i+1+1}]: faktor = {factor1:.6f}")

        # Eliminasi elemen sub-sub-diagonal d[i+2]
        if i + 2 < n:
            factor2 = d[i + 2] / f[i]
            e[i + 2] -= factor2 * g[i]
            f[i + 2] -= factor2 * h[i]
            r[i + 2] -= factor2 * r[i]
            d[i + 2] = 0.0
            print(f"  Elim d[{i+2+1}]: faktor = {factor2:.6f}")

    # Back substitution
    print(f"\n  === Back Substitution ===")
    x = np.zeros(n)
    x[-1] = r[-1] / f[-1]
    print(f"  x[{n}] = {x[-1]:.6f}")

    x[-2] = (r[-2] - g[-2] * x[-1]) / f[-2]
    print(f"  x[{n-1}] = {x[-2]:.6f}")

    for i in range(n - 3, -1, -1):
        x[i] = (r[i] - g[i] * x[i + 1] - h[i] * x[i + 2]) / f[i]
        print(f"  x[{i+1}] = {x[i]:.6f}")

    return x


def extract_bands(A):
    """Ekstrak band dari matriks penuh A (untuk verifikasi)."""
    n = A.shape[0]
    d = np.zeros(n)   # sub-sub diagonal
    e = np.zeros(n)   # sub diagonal
    f = np.diag(A).copy()   # diagonal utama
    g = np.zeros(n)   # super diagonal
    h = np.zeros(n)   # super-super diagonal

    for i in range(2, n):
        d[i] = A[i, i-2]
    for i in range(1, n):
        e[i] = A[i, i-1]
    for i in range(n-1):
        g[i] = A[i, i+1]
    for i in range(n-2):
        h[i] = A[i, i+2]

    return d, e, f, g, h


def main():
    print("=" * 65)
    print("SOAL 11.28 – Solver Sistem Pentadiagonal (Tanpa Pivoting)")
    print("=" * 65)

    # Matriks uji dari buku (Contoh 11.1.1)
    A = np.array([
        [ 8, -2, -1,  0,  0],
        [-2,  9, -4, -1,  0],
        [-1, -3,  7, -1, -2],
        [ 0, -4, -2, 12, -5],
        [ 0,  0, -7, -3, 15]
    ], dtype=float)

    r = np.array([5.0, 2.0, 0.0, 1.0, 5.0])

    print("\nMatriks A (pentadiagonal):")
    print(A)
    print("\nVektor r:", r)

    # Ekstrak band
    d, e, f, g, h = extract_bands(A)
    print(f"\nBand matriks:")
    print(f"  h (super-super) = {h}")
    print(f"  g (super)       = {g}")
    print(f"  f (diagonal)    = {f}")
    print(f"  e (sub)         = {e}")
    print(f"  d (sub-sub)     = {d}")

    # Selesaikan
    x = pentadiagonal_solver(d.copy(), e.copy(), f.copy(), g.copy(), h.copy(), r.copy())

    print(f"\n--- SOLUSI (Pentadiagonal Solver) ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.8f}")

    # Verifikasi dengan numpy
    x_ref = np.linalg.solve(A, r)
    print(f"\n--- Verifikasi (numpy.linalg.solve) ---")
    for i, xi in enumerate(x_ref, 1):
        print(f"  x{i} = {xi:.8f}")

    resid = np.linalg.norm(A @ x - r)
    max_diff = np.max(np.abs(x - x_ref))
    print(f"\n  Norma residu ||Ax - r||   = {resid:.2e}")
    print(f"  Perbedaan maks vs numpy   = {max_diff:.2e}")
    print(f"  {'✓ Solver valid!' if resid < 1e-8 else '✗ Ada perbedaan!'}")

    # Demo struktur pentadiagonal untuk n umum
    print("\n--- Demonstrasi Efisiensi vs Gauss Elimination ---")
    print("  Pentadiagonal: O(n) operasi → sangat efisien untuk n besar")
    print("  Gauss Elim:    O(n³) operasi → mahal untuk n besar")

    ns = [5, 10, 20, 50, 100]
    print(f"\n  {'n':>6} {'Penta (≈8n)':>14} {'Gauss (≈n³/3)':>16}")
    print("  " + "-" * 38)
    for n in ns:
        penta = 8 * n
        gauss = n**3 // 3
        print(f"  {n:>6} {penta:>14,} {gauss:>16,}")


if __name__ == "__main__":
    main()
