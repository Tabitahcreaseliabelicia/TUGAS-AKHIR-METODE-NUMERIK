"""
Soal 11.11 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Gunakan metode Gauss-Seidel untuk menyelesaikan sistem berikut
hingga error persen relatif < εs = 5%:

    10x1 +  2x2 -   x3 =  27
    -3x1 -  6x2 +  2x3 = -61.5
      x1 +   x2 +  5x3 = -21.5
"""

import numpy as np


def check_diagonal_dominance(A):
    """Periksa apakah matriks dominan diagonal."""
    n = A.shape[0]
    dominant = True
    for i in range(n):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag < off:
            dominant = False
    return dominant


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
    print("SOAL 11.11 – Gauss-Seidel: Sistem 3×3 (εs = 5%)")
    print("=" * 65)

    A = np.array([
        [ 10,  2, -1],
        [ -3, -6,  2],
        [  1,  1,  5]
    ], dtype=float)

    b = np.array([27.0, -61.5, -21.5])

    print("\nSistem persamaan:")
    print("  10x1 +  2x2 -   x3 =  27")
    print("  -3x1 -  6x2 +  2x3 = -61.5")
    print("    x1 +   x2 +  5x3 = -21.5")

    # Cek dominansi diagonal
    print("\nPemeriksaan dominansi diagonal:")
    for i in range(3):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(3) if j != i)
        status = "✓ Dominan" if diag > off else "✗ Tidak dominan"
        print(f"  Baris {i+1}: |{A[i,i]:.1f}| vs {off:.1f} → {status}")

    tol = 5.0
    x, history = gauss_seidel(A, b, tol=tol)

    print(f"\n--- Iterasi Gauss-Seidel (εs = {tol}%) ---")
    print(f"{'Iter':<6} {'x1':>12} {'x2':>12} {'x3':>12} {'εs_max':>10}")
    for h in history:
        print(f"{h['iter']:<6} {h['x'][0]:>12.6f} {h['x'][1]:>12.6f} {h['x'][2]:>12.6f} {h['eps']:>10.4f}%")

    print(f"\nKonvergen dalam {len(history)} iterasi.")

    # Solusi eksak
    x_exact = np.linalg.solve(A, b)
    print("\n--- Solusi Eksak (numpy) ---")
    for i, xi in enumerate(x_exact, 1):
        err = abs(x[i-1] - xi) / abs(xi) * 100 if xi != 0 else 0
        print(f"  x{i} = {xi:.6f}  (Gauss-Seidel: {x[i-1]:.6f}, err: {err:.4f}%)")

    # Verifikasi
    resid = np.linalg.norm(A @ x - b)
    print(f"\nNorma residu ||Ax - b|| = {resid:.2e}")


if __name__ == "__main__":
    main()
