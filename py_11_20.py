"""
Soal 11.20 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Ulangi Soal 11.19, tetapi untuk matriks Vandermonde 6-dimensi
(lihat Prob. 10.17) dengan:
  x1 = 4, x2 = 2, x3 = 7, x4 = 10, x5 = 3, x6 = 5

Matriks Vandermonde V[i,j] = x_i^(j-1)  untuk j = 0, 1, ..., n-1

Tentukan spectral condition number dan selesaikan sistemnya.
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def vandermonde_matrix(x):
    """
    Bangun matriks Vandermonde dari vektor x.
    V[i, j] = x[i]^j   (j = 0, 1, ..., n-1)
    """
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = x[i] ** j
    return V


def main():
    print("=" * 65)
    print("SOAL 11.20 – Spectral Condition Number: Matriks Vandermonde 6D")
    print("=" * 65)

    x = np.array([4.0, 2.0, 7.0, 10.0, 3.0, 5.0])
    n = len(x)

    print(f"\nVektor x = {x}")
    print(f"  x1={x[0]}, x2={x[1]}, x3={x[2]}, x4={x[3]}, x5={x[4]}, x6={x[5]}")

    V = vandermonde_matrix(x)

    print(f"\nMatriks Vandermonde {n}×{n}:")
    print("  V[i,j] = x_i^(j-1)")
    for row in V:
        print("  ", [f"{v:.1f}" for v in row])

    # Condition numbers
    cond_2   = np.linalg.cond(V)
    cond_inf = np.linalg.cond(V, np.inf)
    cond_1   = np.linalg.cond(V, 1)

    print(f"\n--- Condition Numbers ---")
    print(f"  Spectral (2-norm):  cond(V) = {cond_2:.4e}")
    print(f"  ∞-norm:             cond(V) = {cond_inf:.4e}")
    print(f"  1-norm:             cond(V) = {cond_1:.4e}")

    digits_lost = math.log10(cond_2) if cond_2 > 0 else 0
    print(f"\n  Digit presisi yang hilang ≈ {digits_lost:.1f}")
    print(f"  (float64 memiliki ~15.9 digit presisi)")

    # Sistem: b = jumlah baris (solusi seharusnya x = 1)
    b = V.sum(axis=1)
    x_sol = np.linalg.solve(V, b)
    x_exact = np.ones(n)
    max_err = np.max(np.abs(x_sol - x_exact))

    print(f"\n--- Solusi Sistem V·c = b (b = jumlah baris, solusi eksak c=1) ---")
    print(f"  Solusi: {np.round(x_sol, 6)}")
    print(f"  Error maks |c - 1| = {max_err:.4e}")

    # Perbandingan dengan Hilbert 6×6
    H6 = np.array([[1/(i+j+1) for j in range(6)] for i in range(6)])
    cond_H6 = np.linalg.cond(H6)

    print(f"\n--- Perbandingan Condition Number ---")
    print(f"  Vandermonde 6×6 :  {cond_2:.4e}")
    print(f"  Hilbert 6×6     :  {cond_H6:.4e}")

    # Plot: Vandermonde vs Hilbert condition numbers untuk berbagai n
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = range(2, 9)
    conds_vdm = []
    conds_hil = []
    for k in sizes:
        xk = np.array([4, 2, 7, 10, 3, 5, 8, 1][:k], dtype=float)
        Vk = vandermonde_matrix(xk)
        Hk = np.array([[1/(i+j+1) for j in range(k)] for i in range(k)])
        conds_vdm.append(np.linalg.cond(Vk))
        conds_hil.append(np.linalg.cond(Hk))

    ax.semilogy(list(sizes), conds_vdm, 'bs-', linewidth=2, label='Vandermonde', markersize=8)
    ax.semilogy(list(sizes), conds_hil, 'r^--', linewidth=2, label='Hilbert', markersize=8)
    ax.axvline(x=6, color='g', linestyle=':', label='n=6')
    ax.set_xlabel('Ukuran matriks n')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_title('Soal 11.20 – Condition Number: Vandermonde vs Hilbert')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('prob_11_20_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_20_plot.png")
    plt.show()

    print("\n--- Kesimpulan ---")
    print(f"  Vandermonde 6D: cond ≈ {cond_2:.2e}")
    print(f"  Semakin besar x dan n, matriks Vandermonde semakin ill-conditioned.")
    print(f"  Ini menyebabkan instabilitas numerik dalam interpolasi polinomial.")


if __name__ == "__main__":
    main()
