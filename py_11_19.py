"""
Soal 11.19 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Gunakan software untuk menentukan spectral condition number
untuk matriks Hilbert 10×10.

Matriks Hilbert: H[i,j] = 1 / (i + j - 1)   (1-indexed)

Pertanyaan:
  - Berapa digit presisi yang diharapkan hilang karena ill-conditioning?
  - Selesaikan sistem untuk kasus b[i] = jumlah koefisien baris i
    (semua x harus = 1).
  - Bandingkan error dengan yang diharapkan dari condition number.
"""

import numpy as np


def hilbert_matrix(n):
    """Buat matriks Hilbert n×n."""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)   # 0-indexed: i+j+1
    return H


def main():
    print("=" * 65)
    print("SOAL 11.19 – Spectral Condition Number: Matriks Hilbert 10×10")
    print("=" * 65)

    # Bangun matriks Hilbert 10×10
    n = 10
    H = hilbert_matrix(n)

    print(f"\nMatriks Hilbert {n}×{n}:")
    print("  H[i,j] = 1/(i+j-1)  (indeks 1-based)")
    print(f"\n  Elemen H[1,1] = {H[0,0]:.4f}")
    print(f"  Elemen H[1,2] = {H[0,1]:.4f}")
    print(f"  Elemen H[n,n] = {H[-1,-1]:.6f}")

    # Condition numbers
    cond_2    = np.linalg.cond(H)           # spectral (2-norm)
    cond_inf  = np.linalg.cond(H, np.inf)   # ∞-norm
    cond_1    = np.linalg.cond(H, 1)        # 1-norm

    print(f"\n--- Condition Numbers ---")
    print(f"  Spectral (2-norm):  cond(H) = {cond_2:.4e}")
    print(f"  ∞-norm:             cond(H) = {cond_inf:.4e}")
    print(f"  1-norm:             cond(H) = {cond_1:.4e}")

    # Estimasi digit yang hilang
    import math
    digits_lost_2   = math.log10(cond_2)
    digits_lost_inf = math.log10(cond_inf)
    machine_eps = np.finfo(float).eps
    print(f"\n  Machine epsilon: {machine_eps:.2e} (≈ {-math.log10(machine_eps):.1f} digit presisi)")
    print(f"\n  Digit presisi yang HILANG:")
    print(f"    Berdasarkan 2-norm:  log10({cond_2:.2e}) ≈ {digits_lost_2:.1f} digit")
    print(f"    Berdasarkan ∞-norm:  log10({cond_inf:.2e}) ≈ {digits_lost_inf:.1f} digit")

    # Sistem dengan b = jumlah tiap baris (solusi seharusnya semua x=1)
    b = H.sum(axis=1)   # b[i] = sum of row i
    x = np.linalg.solve(H, b)

    print(f"\n--- Solusi Sistem H·x = b (b = jumlah baris, solusi eksak x=1) ---")
    x_exact = np.ones(n)
    max_err = np.max(np.abs(x - x_exact))
    rel_err = max_err / np.max(np.abs(x_exact))
    print(f"  Error maks |x_computed - 1| = {max_err:.4e}")
    print(f"  Error relatif maks           = {rel_err:.4e}")
    print(f"  log10(error rel)             ≈ {math.log10(rel_err):.1f} digit")

    print(f"\n  x[1..5] = {x[:5]}")
    print(f"  x[6..10] = {x[5:]}")

    # Plot kondisi
    import matplotlib.pyplot as plt
    sizes   = range(2, 14)
    cond_vals = []
    for k in sizes:
        Hk = hilbert_matrix(k)
        cond_vals.append(np.linalg.cond(Hk))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(list(sizes), cond_vals, 'bo-', linewidth=2, markersize=7)
    ax.axvline(x=10, color='r', linestyle='--', label='n=10')
    ax.set_xlabel('Ukuran matriks n')
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_title('Soal 11.19 – Condition Number Matriks Hilbert')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('prob_11_19_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_19_plot.png")
    plt.show()

    print("\n--- Kesimpulan ---")
    print(f"  Untuk matriks Hilbert 10×10:")
    print(f"  cond ≈ {cond_2:.2e} → hilang ~{digits_lost_2:.0f} digit presisi")
    print(f"  Dengan 15-16 digit presisi (float64), sisa ~{16-digits_lost_2:.0f} digit akurat.")


if __name__ == "__main__":
    main()
