"""
Soal 11.4 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Konfirmasi validitas dekomposisi Cholesky dari Contoh 11.2 dengan
memverifikasi bahwa [L][L]^T = [A].

Contoh 11.2 — Sistem simetris (dari buku):
    A = [6   15   55]
        [15  55  225]
        [55 225  979]

Dekomposisi Cholesky: A = L @ L^T
Verifikasi: L @ L^T harus menghasilkan kembali A.
"""

import numpy as np


def cholesky_decompose(A):
    """
    Dekomposisi Cholesky manual: A = L @ L^T
    Hanya berlaku untuk matriks simetris positif-definit.

    Return:
        L : matriks segitiga bawah
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError("Matriks tidak positif-definit!")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def main():
    print("=" * 60)
    print("SOAL 11.4 – Verifikasi Dekomposisi Cholesky: L @ Lᵀ = A")
    print("=" * 60)

    # Contoh 11.2 dari buku
    A = np.array([
        [  6,  15,   55],
        [ 15,  55,  225],
        [ 55, 225,  979]
    ], dtype=float)

    print("\nMatriks A (Contoh 11.2):")
    print(A)

    # Dekomposisi Cholesky manual
    L = cholesky_decompose(A)

    print("\n--- Matriks L (segitiga bawah) ---")
    print(np.round(L, 6))

    print("\n--- Matriks Lᵀ (segitiga atas) ---")
    print(np.round(L.T, 6))

    # Verifikasi: L @ L^T harus = A
    product = L @ L.T

    print("\n--- Produk L × Lᵀ ---")
    print(np.round(product, 6))

    print("\n--- Matriks A asli ---")
    print(A)

    valid = np.allclose(product, A)
    print(f"\n{'✓ L × Lᵀ = A (terverifikasi!)' if valid else '✗ GAGAL!'}")

    # Bandingkan dengan numpy Cholesky
    L_np = np.linalg.cholesky(A)
    print("\n--- Referensi: numpy.linalg.cholesky(A) ---")
    print(np.round(L_np, 6))

    diff = np.max(np.abs(L - L_np))
    print(f"\nPerbedaan maks dengan numpy: {diff:.2e}")


if __name__ == "__main__":
    main()
