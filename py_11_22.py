"""
Soal 11.22 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Tulis himpunan persamaan berikut dalam bentuk matriks:

    50 = 5x3 - 7x2
    4x2 + 7x3 + 30 = 0
    x1 - 7x3 = 40 - 3x2 + 5x1

Gunakan Excel, MATLAB, atau Mathcad untuk menyelesaikan unknowns.
Selain itu, hitung transpose dan invers matriks koefisien.
"""

import numpy as np


def main():
    print("=" * 65)
    print("SOAL 11.22 – Bentuk Matriks + Transpose + Invers")
    print("=" * 65)

    print("\nPersamaan asli:")
    print("  50 = 5x3 - 7x2                  →  0·x1 - 7x2 + 5x3 = 50")
    print("  4x2 + 7x3 + 30 = 0              →  0·x1 + 4x2 + 7x3 = -30")
    print("  x1 - 7x3 = 40 - 3x2 + 5x1      →  -4x1 + 3x2 - 7x3 = 40")

    print("\nPenyusunan ulang (bentuk standar Ax = b):")
    print("  Pers. 1:  0·x1  - 7x2  + 5x3  = 50")
    print("  Pers. 2:  0·x1  + 4x2  + 7x3  = -30")
    print("  Pers. 3: -4·x1  + 3x2  - 7x3  = 40")

    # Matriks koefisien setelah penyusunan ulang
    A = np.array([
        [ 0, -7,  5],
        [ 0,  4,  7],
        [-4,  3, -7]
    ], dtype=float)

    b = np.array([50.0, -30.0, 40.0])

    print(f"\n--- Matriks A ---")
    print(A)

    print(f"\n--- Vektor b ---")
    print(b)

    # Solusi
    x = np.linalg.solve(A, b)
    print(f"\n--- Solusi ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.6f}")

    # Verifikasi
    resid = np.linalg.norm(A @ x - b)
    print(f"\n  Norma residu ||Ax - b|| = {resid:.2e}")

    # Transpose
    A_T = A.T
    print(f"\n--- Transpose Aᵀ ---")
    print(A_T)

    print(f"\n  Properti Transpose:")
    print(f"  (Aᵀ)ᵀ = A? {np.allclose(A_T.T, A)}")
    print(f"  (AB)ᵀ = BᵀAᵀ? (perlu B, skip)")

    # Invers
    try:
        A_inv = np.linalg.inv(A)
        print(f"\n--- Invers A⁻¹ ---")
        print(np.round(A_inv, 6))

        print(f"\n  Verifikasi A × A⁻¹:")
        print(np.round(A @ A_inv, 6))

        # Solusi via invers
        x_inv = A_inv @ b
        print(f"\n--- Solusi via x = A⁻¹ · b ---")
        for i, xi in enumerate(x_inv, 1):
            print(f"  x{i} = {xi:.6f}")

        # Condition number
        cond = np.linalg.cond(A)
        print(f"\n--- Condition Number ---")
        print(f"  cond(A) = {cond:.4f}")

    except np.linalg.LinAlgError:
        print("\n⚠ Matriks singular! Tidak dapat dihitung inversnya.")
        print("  Periksa apakah sistem memiliki solusi unik.")


if __name__ == "__main__":
    main()
