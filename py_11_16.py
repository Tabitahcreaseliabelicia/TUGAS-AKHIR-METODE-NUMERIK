"""
Soal 11.16 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Gunakan paket perangkat lunak untuk: mendapatkan solusi, menghitung invers,
dan menentukan condition number (berdasarkan norma row-sum / ∞-norm)
untuk:

(a) Sistem 3×3:
    [1  4  9] [x1]   [14]
    [4  9 16] [x2] = [29]
    [9 16 25] [x3]   [50]

(b) Sistem 4×4:
    [1  4  9  16] [x1]   [ 30]
    [4  9 16  25] [x2] = [ 54]
    [9 16 25  36] [x3]   [ 86]
    [16 25 36 49] [x4]   [126]

Semua nilai x harus = 1.
"""

import numpy as np


def row_sum_norm(A):
    """Norma baris-sum (∞-norm) matriks."""
    return np.max(np.sum(np.abs(A), axis=1))


def condition_number_row_sum(A):
    """Condition number berbasis norma row-sum: cond = ||A|| × ||A⁻¹||."""
    A_inv = np.linalg.inv(A)
    return row_sum_norm(A) * row_sum_norm(A_inv)


def solve_and_analyze(A, b, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    n = A.shape[0]

    print("\nMatriks A:")
    print(A)
    print("\nVektor b:", b)

    # Solusi
    x = np.linalg.solve(A, b)
    print("\n--- Solusi ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.8f}")

    # Verifikasi (semua harus = 1)
    expected = np.ones(n)
    err = np.max(np.abs(x - expected))
    print(f"\n  (Semua x harus = 1, error maks: {err:.2e})")

    # Invers
    A_inv = np.linalg.inv(A)
    print("\n--- Invers A ---")
    print(np.round(A_inv, 6))

    # Verifikasi invers
    I_check = A @ A_inv
    print("\n--- A × A⁻¹ (harus ≈ I) ---")
    print(np.round(I_check, 6))

    # Condition number (row-sum norm)
    norm_A     = row_sum_norm(A)
    norm_A_inv = row_sum_norm(A_inv)
    cond_rs    = condition_number_row_sum(A)
    cond_np    = np.linalg.cond(A, np.inf)  # ∞-norm

    print("\n--- Condition Number ---")
    print(f"  ||A||∞        = {norm_A:.6f}")
    print(f"  ||A⁻¹||∞      = {norm_A_inv:.6f}")
    print(f"  Cond (row-sum) = {cond_rs:.4f}")
    print(f"  Cond (numpy ∞) = {cond_np:.4f}")

    # Interpretasi
    if cond_rs > 1e6:
        print("  ⚠ Matriks sangat ill-conditioned!")
    elif cond_rs > 100:
        print("  ⚠ Matriks cukup ill-conditioned.")
    else:
        print("  ✓ Matriks well-conditioned.")

    return cond_rs


def main():
    print("=" * 60)
    print("SOAL 11.16 – Condition Number & Invers (Norma Row-Sum)")
    print("=" * 60)

    # (a) 3×3
    A3 = np.array([
        [1,  4,  9],
        [4,  9, 16],
        [9, 16, 25]
    ], dtype=float)
    b3 = np.array([14.0, 29.0, 50.0])

    # (b) 4×4
    A4 = np.array([
        [ 1,  4,  9, 16],
        [ 4,  9, 16, 25],
        [ 9, 16, 25, 36],
        [16, 25, 36, 49]
    ], dtype=float)
    b4 = np.array([30.0, 54.0, 86.0, 126.0])

    c3 = solve_and_analyze(A3, b3, "(a) Sistem 3×3")
    c4 = solve_and_analyze(A4, b4, "(b) Sistem 4×4")

    print(f"\n{'='*55}")
    print(f"RINGKASAN:")
    print(f"  (a) 3×3 → Cond = {c3:.4f}")
    print(f"  (b) 4×4 → Cond = {c4:.4f}")
    print("\nMatriks ini merupakan sub-matriks dari Hilbert-like matrix,")
    print("yang cenderung semakin ill-conditioned seiring pertambahan ukuran.")


if __name__ == "__main__":
    main()
