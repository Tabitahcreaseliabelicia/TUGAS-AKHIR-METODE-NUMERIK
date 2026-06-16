"""
Soal 11.10 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Ulangi Soal 11.9, tetapi gunakan iterasi Jacobi.

Sistem (sama dengan 11.9):
    15c1 -  3c2 -   c3 = 3800
    -3c1 + 18c2 -  6c3 = 1200
    -4c1 -   c2 + 12c3 = 2350

Perbedaan Jacobi vs Gauss-Seidel:
  - Jacobi  : semua nilai baru dihitung BERSAMAAN dari nilai lama
  - Gauss-Seidel: nilai baru langsung dipakai dalam iterasi yang sama
"""

import numpy as np


def jacobi_iteration(A, b, x0=None, tol=5.0, max_iter=100):
    """
    Iterasi Jacobi untuk sistem linear Ax = b.

    Parameter:
        A       : matriks koefisien
        b       : vektor RHS
        x0      : tebakan awal
        tol     : toleransi persen relatif (%)
        max_iter: iterasi maksimum

    Return:
        x       : solusi
        history : riwayat iterasi
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        x_new = np.zeros(n)

        # SEMUA dihitung dari x_old (bukan x yang sedang diupdate)
        for i in range(n):
            sigma = sum(A[i, j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i, i]

        x = x_new.copy()

        # Error
        eps = [abs((x[i] - x_old[i]) / x[i]) * 100 for i in range(n) if x[i] != 0]
        max_eps = max(eps) if eps else 0
        history.append({'iter': iteration, 'x': x.copy(), 'eps': max_eps})

        if max_eps < tol:
            break

    return x, history


def gauss_seidel(A, b, tol=5.0, max_iter=100):
    """Gauss-Seidel untuk perbandingan."""
    n = len(b)
    x = np.zeros(n)
    history = []
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        eps = [abs((x[i] - x_old[i]) / x[i]) * 100 for i in range(n) if x[i] != 0]
        max_eps = max(eps) if eps else 0
        history.append({'iter': iteration, 'x': x.copy(), 'eps': max_eps})
        if max_eps < tol:
            break
    return x, history


def main():
    print("=" * 65)
    print("SOAL 11.10 – Iterasi Jacobi untuk Sistem Coupled Reactors")
    print("=" * 65)

    A = np.array([
        [ 15,  -3,  -1],
        [ -3,  18,  -6],
        [ -4,  -1,  12]
    ], dtype=float)
    b = np.array([3800.0, 1200.0, 2350.0])

    print("\nSistem persamaan:")
    print("  15c1 -  3c2 -   c3 = 3800")
    print("  -3c1 + 18c2 -  6c3 = 1200")
    print("  -4c1 -   c2 + 12c3 = 2350")

    tol = 5.0

    # Jacobi
    x_j, hist_j = jacobi_iteration(A, b, tol=tol)
    print(f"\n--- Iterasi Jacobi (εs = {tol}%) ---")
    print(f"{'Iter':<6} {'c1':>12} {'c2':>12} {'c3':>12} {'εs_max':>10}")
    for h in hist_j:
        print(f"{h['iter']:<6} {h['x'][0]:>12.4f} {h['x'][1]:>12.4f} {h['x'][2]:>12.4f} {h['eps']:>10.4f}%")

    # Gauss-Seidel untuk perbandingan
    x_gs, hist_gs = gauss_seidel(A, b, tol=tol)
    print(f"\n--- Gauss-Seidel (perbandingan, εs = {tol}%) ---")
    print(f"{'Iter':<6} {'c1':>12} {'c2':>12} {'c3':>12} {'εs_max':>10}")
    for h in hist_gs:
        print(f"{h['iter']:<6} {h['x'][0]:>12.4f} {h['x'][1]:>12.4f} {h['x'][2]:>12.4f} {h['eps']:>10.4f}%")

    # Solusi eksak
    x_exact = np.linalg.solve(A, b)
    print("\n--- Solusi Eksak ---")
    labels = ['c1', 'c2', 'c3']
    for i, ci in enumerate(x_exact):
        print(f"  {labels[i]} = {ci:.6f}")

    print(f"\n📊 Perbandingan: Jacobi = {len(hist_j)} iterasi, Gauss-Seidel = {len(hist_gs)} iterasi")
    print("   Gauss-Seidel biasanya lebih cepat konvergen daripada Jacobi.")


if __name__ == "__main__":
    main()
