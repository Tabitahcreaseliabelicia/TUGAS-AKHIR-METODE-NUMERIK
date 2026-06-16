"""
Soal 11.13 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Gunakan metode Gauss-Seidel (a) tanpa relaxation dan
(b) dengan relaxation λ = 1.2 untuk menyelesaikan sistem berikut
dengan toleransi εs = 5%:

    2x1 -  6x2 -   x3 = -38
   -3x1 -   x2 +  7x3 = -34
   -8x1 +   x2 -  2x3 = -20

Jika perlu, susun ulang persamaan untuk mencapai konvergensi.
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=5.0, max_iter=200, lam=1.0):
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
    print("=" * 70)
    print("SOAL 11.13 – Gauss-Seidel Tanpa & Dengan Overrelaxation (λ=1.2)")
    print("=" * 70)

    # Sistem asli
    A_orig = np.array([
        [ 2, -6, -1],
        [-3, -1,  7],
        [-8,  1, -2]
    ], dtype=float)
    b_orig = np.array([-38.0, -34.0, -20.0])

    print("\nSistem asli:")
    print("   2x1 -  6x2 -   x3 = -38")
    print("  -3x1 -   x2 +  7x3 = -34")
    print("  -8x1 +   x2 -  2x3 = -20")

    # Cek dominansi diagonal
    print("\nPemeriksaan dominansi diagonal (ASLI):")
    for i in range(3):
        diag = abs(A_orig[i, i])
        off  = sum(abs(A_orig[i, j]) for j in range(3) if j != i)
        status = "✓ Dominan" if diag > off else "✗ TIDAK dominan"
        print(f"  Baris {i+1}: |{A_orig[i,i]:.0f}| vs {off:.0f} → {status}")

    # Susun ulang untuk dominansi diagonal:
    # Pilih baris dengan elemen terbesar di kolom masing-masing sebagai diagonal
    # Baris 3 → isolasi x1: -8x1 + x2 - 2x3 = -20  → |8| > |3|, dominan utk x1
    # Baris 1 → isolasi x2: 2x1 - 6x2 - x3 = -38   → |6| > |3|, dominan utk x2
    # Baris 2 → isolasi x3: -3x1 - x2 + 7x3 = -34  → |7| > |4|, dominan utk x3
    A = np.array([
        [-8,  1, -2],   # baris 3 → utk x1
        [ 2, -6, -1],   # baris 1 → utk x2
        [-3, -1,  7]    # baris 2 → utk x3
    ], dtype=float)
    b = np.array([-20.0, -38.0, -34.0])

    print("\nSistem SETELAH disusun ulang (agar dominan diagonal):")
    print("  -8x1 +   x2 -  2x3 = -20  → isolasi x1")
    print("   2x1 -  6x2 -   x3 = -38  → isolasi x2")
    print("  -3x1 -   x2 +  7x3 = -34  → isolasi x3")

    print("\nPemeriksaan dominansi diagonal (SETELAH):")
    for i in range(3):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(3) if j != i)
        status = "✓ Dominan" if diag > off else "✗ Tidak dominan"
        print(f"  Baris {i+1}: |{A[i,i]:.0f}| vs {off:.0f} → {status}")

    tol = 5.0

    # (a) Tanpa relaxation
    print(f"\n(a) Gauss-Seidel TANPA relaxation (λ=1.0)")
    x_a, hist_a = gauss_seidel(A, b, tol=tol, lam=1.0)
    print(f"{'Iter':<6} {'x1':>12} {'x2':>12} {'x3':>12} {'εs_max':>10}")
    for h in hist_a[:20]:
        print(f"{h['iter']:<6} {h['x'][0]:>12.6f} {h['x'][1]:>12.6f} {h['x'][2]:>12.6f} {h['eps']:>10.4f}%")
    if len(hist_a) > 20:
        print(f"  ... ({len(hist_a)} total iterasi)")

    # (b) Dengan overrelaxation λ=1.2
    lam = 1.2
    print(f"\n(b) Gauss-Seidel DENGAN Overrelaxation (λ={lam})")
    x_b, hist_b = gauss_seidel(A, b, tol=tol, lam=lam)
    print(f"{'Iter':<6} {'x1':>12} {'x2':>12} {'x3':>12} {'εs_max':>10}")
    for h in hist_b[:20]:
        print(f"{h['iter']:<6} {h['x'][0]:>12.6f} {h['x'][1]:>12.6f} {h['x'][2]:>12.6f} {h['eps']:>10.4f}%")
    if len(hist_b) > 20:
        print(f"  ... ({len(hist_b)} total iterasi)")

    # Solusi eksak
    x_exact = np.linalg.solve(A_orig, b_orig)
    print("\n--- Solusi Eksak (numpy) ---")
    for i, xi in enumerate(x_exact, 1):
        print(f"  x{i} = {xi:.6f}")

    print(f"\n📊 Tanpa relaxation: {len(hist_a)} iterasi")
    print(f"   Dengan λ={lam}:     {len(hist_b)} iterasi")


if __name__ == "__main__":
    main()
