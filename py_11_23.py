"""
Soal 11.23 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Dalam Sec. 9.2.1, jumlah operasi untuk eliminasi Gauss tanpa
partial pivoting ditentukan. Lakukan hal serupa untuk Thomas Algorithm.
Buat plot jumlah operasi vs n (dari n=2 hingga n=20) untuk
kedua teknik tersebut.

Hasil teoritis:
  - Eliminasi Gauss:   O(n³/3)  → lebih tepatnya  (2n³ + 3n² - 5n)/6
  - Thomas Algorithm:  O(n)     → lebih tepatnya  8(n-1) operasi total
"""

import numpy as np
import matplotlib.pyplot as plt


def ops_gauss_elimination(n):
    """
    Jumlah operasi (perkalian + pembagian) untuk eliminasi Gauss
    tanpa partial pivoting.

    Forward elimination: Σ_{k=1}^{n-1} Σ_{i=k+1}^{n} (n-k+2)
    ≈ n³/3 + n²/2 - 5n/6  (perkalian/pembagian dominan)

    Rumus dari buku Chapra (Eq. 9.12):
      Perkalian & Pembagian = (2n³ + 3n² - 5n) / 6
      Penjumlahan & Pengurangan = (n³ - n) / 3  ← tidak kita hitung disini
    """
    # Rumus eksak untuk operasi perkalian/pembagian
    mult_div = (2 * n**3 + 3 * n**2 - 5 * n) / 6
    add_sub  = (n**3 - n) / 3
    return int(mult_div + add_sub)


def ops_thomas_algorithm(n):
    """
    Jumlah operasi untuk Thomas Algorithm (tridiagonal n×n).

    Forward sweep: (n-1) pembagian + (n-1) perkalian + (n-1) pengurangan
                 = 3(n-1) operasi
    Back substitution: 1 pembagian + (n-1) perkalian + (n-1) pengurangan
                      = 2(n-1) + 1 ≈ 2(n-1)
    Total ≈ 5(n-1) operasi aritmatik dominan

    Atau dalam hitungan semua operasi:
      Forward: 2(n-1) mult/div + (n-1) add/sub = 3(n-1)
      RHS fwd: (n-1) mult + (n-1) sub = 2(n-1)
      Back sub: (n-1) mult + (n-1) sub + 1 div = 2(n-1)+1
    Total ≈ 7(n-1) + 1
    """
    return 7 * (n - 1) + 1


def count_exact_operations():
    """
    Hitung operasi EKSAK dengan mensimulasikan Thomas Algorithm.
    """
    results = {}
    for n in range(2, 21):
        ops = 0
        # Forward sweep
        for i in range(1, n):
            ops += 1   # pembagian: factor = a[i] / b[i-1]
            ops += 1   # perkalian: factor * c[i-1]
            ops += 1   # pengurangan: b[i] -= ...
            ops += 1   # perkalian: factor * d[i-1]
            ops += 1   # pengurangan: d[i] -= ...
        # Back substitution
        ops += 1       # pembagian: x[-1] = d[-1]/b[-1]
        for i in range(n - 2, -1, -1):
            ops += 1   # perkalian: c[i] * x[i+1]
            ops += 1   # pengurangan: d[i] - ...
            ops += 1   # pembagian: / b[i]
        results[n] = ops
    return results


def main():
    print("=" * 65)
    print("SOAL 11.23 – Jumlah Operasi: Gauss Elimination vs Thomas Algorithm")
    print("=" * 65)

    n_vals = list(range(2, 21))

    gauss_ops  = [ops_gauss_elimination(n) for n in n_vals]
    thomas_ops = [ops_thomas_algorithm(n) for n in n_vals]
    thomas_exact = count_exact_operations()

    print(f"\n{'n':>5} {'Gauss Elim':>15} {'Thomas (formula)':>18} {'Thomas (eksak)':>16}")
    print("-" * 58)
    for n, g, t in zip(n_vals, gauss_ops, thomas_ops):
        te = thomas_exact[n]
        print(f"{n:>5} {g:>15,} {t:>18,} {te:>16,}")

    # Plot perbandingan
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Soal 11.23 – Jumlah Operasi Aritmatik", fontsize=14, fontweight='bold')

    # Plot 1: Linear scale
    ax1 = axes[0]
    ax1.plot(n_vals, gauss_ops,  'rs-', linewidth=2, markersize=7, label='Gauss Elimination O(n³)')
    ax1.plot(n_vals, thomas_ops, 'b^-', linewidth=2, markersize=7, label='Thomas Algorithm O(n)')
    ax1.plot(n_vals, [thomas_exact[n] for n in n_vals], 'g--',
             linewidth=1.5, label='Thomas (eksak)')
    ax1.set_xlabel('Ukuran sistem n')
    ax1.set_ylabel('Jumlah Operasi')
    ax1.set_title('Skala Linear')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log scale
    ax2 = axes[1]
    ax2.semilogy(n_vals, gauss_ops,  'rs-', linewidth=2, markersize=7, label='Gauss Elimination O(n³)')
    ax2.semilogy(n_vals, thomas_ops, 'b^-', linewidth=2, markersize=7, label='Thomas Algorithm O(n)')
    ax2.set_xlabel('Ukuran sistem n')
    ax2.set_ylabel('Jumlah Operasi (log scale)')
    ax2.set_title('Skala Logaritmik')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('prob_11_23_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_23_plot.png")
    plt.show()

    print("\n--- Analisis ---")
    print("  Thomas Algorithm jauh lebih efisien untuk sistem tridiagonal:")
    for n in [5, 10, 20]:
        g = ops_gauss_elimination(n)
        t = thomas_exact[n]
        ratio = g / t
        print(f"  n={n:2d}: Gauss={g:6,}, Thomas={t:3}, rasio={ratio:.1f}×")

    print("\n  Gauss Elimination:  O(n³) → tidak praktis untuk n besar")
    print("  Thomas Algorithm:   O(n)  → linier, sangat efisien")


if __name__ == "__main__":
    main()
