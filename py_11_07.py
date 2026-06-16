"""
Soal 11.7 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Hitung dekomposisi Cholesky untuk:

    [A] = [9  0  0]
          [0 25  0]
          [0  0  4]

Apakah hasilnya masuk akal dalam konteks persamaan (11.3) dan (11.4)?

Catatan:
  Untuk matriks diagonal, L harus berupa matriks diagonal dengan
  l_ii = sqrt(a_ii), sehingga L @ L^T = A secara langsung.
"""

import numpy as np
import math


def cholesky_decompose(A):
    """Dekomposisi Cholesky: A = L @ L^T."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val < 0:
                    raise ValueError("Matriks tidak positif semi-definit!")
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def main():
    print("=" * 60)
    print("SOAL 11.7 – Cholesky Decomposition untuk Matriks Diagonal")
    print("=" * 60)

    A = np.array([
        [9, 0, 0],
        [0, 25, 0],
        [0, 0, 4]
    ], dtype=float)

    print("\nMatriks A:")
    print(A)

    print("\nUntuk matriks diagonal, teoritis L_ii = sqrt(A_ii):")
    for i in range(3):
        print(f"  l_{i+1}{i+1} = sqrt({A[i,i]:.0f}) = {math.sqrt(A[i,i]):.6f}")

    L = cholesky_decompose(A)

    print("\n--- Hasil Cholesky Decomposition ---")
    print("Matriks L:")
    print(np.round(L, 6))

    print("\n--- Verifikasi L × Lᵀ ---")
    product = L @ L.T
    print(np.round(product, 6))

    valid = np.allclose(product, A)
    print(f"\n{'✓ L × Lᵀ = A (benar)' if valid else '✗ GAGAL!'}")

    print("\n--- Analisis ---")
    print("Untuk matriks diagonal positif-definit D:")
    print("  L = sqrt(D)  (matriks diagonal dengan elemen akar kuadrat)")
    print("  Hal ini konsisten dengan persamaan (11.3) dan (11.4).")
    print("  Karena semua elemen off-diagonal = 0,")
    print("  tidak ada coupling antar variabel.")
    print(f"\n  l_11 = sqrt(9)  = {math.sqrt(9):.4f}")
    print(f"  l_22 = sqrt(25) = {math.sqrt(25):.4f}")
    print(f"  l_33 = sqrt(4)  = {math.sqrt(4):.4f}")
    print("\n  Hasil masuk akal: Cholesky pada matriks diagonal")
    print("  hanya memerlukan operasi sqrt, tanpa eliminasi.")


if __name__ == "__main__":
    main()
