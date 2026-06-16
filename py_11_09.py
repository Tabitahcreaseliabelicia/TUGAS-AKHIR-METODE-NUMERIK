"""
Soal 11.9 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Gunakan metode Gauss-Seidel (εs = 5%) untuk menyelesaikan sistem
coupled reactor berikut:

    15c1 -  3c2 -   c3 = 3800
    -3c1 + 18c2 -  6c3 = 1200
    -4c1 -   c2 + 12c3 = 2350

(konsentrasi dalam g/m³)
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=5.0, max_iter=100, lam=1.0):
    """Gauss-Seidel dengan relaxation opsional."""
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]

        eps = [abs((x[i] - x_old[i]) / x[i]) * 100 for i in range(n) if x[i] != 0]
        max_eps = max(eps) if eps else 0
        history.append({'iter': iteration, 'x': x.copy(), 'eps': max_eps})

        if max_eps < tol:
            break

    return x, history


def main():
    print("=" * 65)
    print("SOAL 11.9 – Gauss-Seidel untuk Sistem Coupled Reactors")
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

    # Cek dominansi diagonal
    print("\nPemeriksaan dominansi diagonal:")
    for i in range(3):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(3) if j != i)
        status = "✓ Dominan" if diag > off else "✗ Tidak dominan"
        print(f"  Baris {i+1}: |{A[i,i]:.0f}| > {off:.0f} → {status}")

    tol = 5.0
    x, history = gauss_seidel(A, b, tol=tol)

    print(f"\n--- Iterasi Gauss-Seidel (εs = {tol}%) ---")
    print(f"{'Iter':<6} {'c1':>12} {'c2':>12} {'c3':>12} {'εs_max':>10}")
    for h in history:
        print(f"{h['iter']:<6} {h['x'][0]:>12.4f} {h['x'][1]:>12.4f} {h['x'][2]:>12.4f} {h['eps']:>10.4f}%")

    print(f"\nKonvergen dalam {len(history)} iterasi.")

    # Verifikasi
    x_exact = np.linalg.solve(A, b)
    print("\n--- Solusi Eksak (numpy) ---")
    labels = ['c1', 'c2', 'c3']
    for i, (ci, ci_ref) in enumerate(zip(x, x_exact)):
        err = abs(ci - ci_ref) / ci_ref * 100
        print(f"  {labels[i]} = {ci:.4f}  (eksak: {ci_ref:.4f}, err: {err:.4f}%)")


if __name__ == "__main__":
    main()
