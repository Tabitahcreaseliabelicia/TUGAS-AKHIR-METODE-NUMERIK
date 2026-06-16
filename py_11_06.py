"""
Soal 11.6 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Lakukan dekomposisi Cholesky dengan tangan (secara manual, langkah per langkah)
untuk sistem simetris berikut:

    [ 8   20   15] [x1]   [ 50]
    [20   80   50] [x2] = [250]
    [15   50   60] [x3]   [100]

Tunjukkan setiap langkah perhitungan elemen L secara eksplisit.
"""

import numpy as np
import math


def cholesky_manual_verbose(A):
    """
    Dekomposisi Cholesky dengan tampilan langkah-langkah manual.
    Menampilkan setiap formula dan hasil perhitungan.
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    print("\n=== LANGKAH-LANGKAH DEKOMPOSISI CHOLESKY ===")
    print("Formula: l_ij = (a_ij - Σ l_ik·l_jk) / l_jj  (i≠j)")
    print("         l_ii = √(a_ii - Σ l_ik²)")

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))

            if i == j:
                val = A[i, i] - s
                L[i, j] = math.sqrt(val)
                if j == 0:
                    print(f"\n  l_{i+1}{j+1} = √(a_{i+1}{j+1}) = √({A[i,i]:.4f}) = {L[i,j]:.6f}")
                else:
                    terms = " + ".join([f"l_{i+1}{k+1}²" for k in range(j)])
                    print(f"\n  l_{i+1}{j+1} = √(a_{i+1}{j+1} - ({terms})) = √({val:.4f}) = {L[i,j]:.6f}")
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                if j == 0:
                    print(f"  l_{i+1}{j+1} = a_{i+1}{j+1} / l_{j+1}{j+1} = {A[i,j]:.4f} / {L[j,j]:.6f} = {L[i,j]:.6f}")
                else:
                    terms = " + ".join([f"l_{i+1}{k+1}·l_{j+1}{k+1}" for k in range(j)])
                    print(f"  l_{i+1}{j+1} = (a_{i+1}{j+1} - ({terms})) / l_{j+1}{j+1} = {L[i,j]:.6f}")

    return L


def forward_sub(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_sub(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def main():
    print("=" * 60)
    print("SOAL 11.6 – Dekomposisi Cholesky Manual (3×3)")
    print("=" * 60)

    A = np.array([
        [ 8,  20,  15],
        [20,  80,  50],
        [15,  50,  60]
    ], dtype=float)

    b = np.array([50.0, 250.0, 100.0])

    print("\nMatriks A:")
    for row in A:
        print("  ", row)

    print("\nVektor b:", b)

    # Cek positif-definit
    eigenvals = np.linalg.eigvalsh(A)
    print(f"\nEigenvalue: {np.round(eigenvals, 4)}")
    if all(eigenvals > 0):
        print("✓ Matriks positif-definit, Cholesky dapat diterapkan.")
    else:
        print("✗ Matriks tidak positif-definit!")
        return

    # Dekomposisi manual dengan tampilan langkah
    L = cholesky_manual_verbose(A)

    print("\n\n--- Matriks L (Segitiga Bawah) ---")
    print(np.round(L, 6))

    print("\n--- Matriks Lᵀ (Segitiga Atas) ---")
    print(np.round(L.T, 6))

    # Verifikasi L @ L^T = A
    print("\n--- Verifikasi L × Lᵀ ---")
    print(np.round(L @ L.T, 6))
    print(f"{'✓ L × Lᵀ = A' if np.allclose(L @ L.T, A) else '✗ GAGAL'}")

    # Selesaikan sistem
    y = forward_sub(L, b)
    x = back_sub(L.T, y)

    print("\n--- Solusi Sistem ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.6f}")

    resid = np.linalg.norm(A @ x - b)
    print(f"\nNorma residu: {resid:.2e}")

    # Referensi
    x_ref = np.linalg.solve(A, b)
    print("\n--- Referensi (numpy) ---")
    for i, xi in enumerate(x_ref, 1):
        print(f"  x{i} = {xi:.6f}")


if __name__ == "__main__":
    main()
