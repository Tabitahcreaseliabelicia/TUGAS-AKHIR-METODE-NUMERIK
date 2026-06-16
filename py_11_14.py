"""
Soal 11.14 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Gambar ulang Fig. 11.5 untuk kasus di mana kemiringan (slope) persamaan-
persamaan adalah 1 dan -1. Apa yang terjadi ketika Gauss-Seidel diterapkan
pada sistem tersebut?

Analisis:
  Jika kedua persamaan memiliki slope 1 dan -1 (garis berpotongan tegak lurus),
  metode Gauss-Seidel TIDAK menjamin konvergensi karena:
  - Matriks tidak memenuhi syarat dominansi diagonal ketat
  - Iterasi Gauss-Seidel dapat berosilasi atau divergen

Visualisasi dilakukan dengan matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def gauss_seidel_2d(A, b, x0=None, max_iter=20, lam=1.0):
    """Gauss-Seidel 2D dengan riwayat lengkap."""
    x = np.array(x0 if x0 else [0.0, 0.0], dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(2):
            j = 1 - i
            x_new = (b[i] - A[i, j] * x[j]) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]
        path.append(x.copy())
    return path


def main():
    print("=" * 65)
    print("SOAL 11.14 – Divergensi Gauss-Seidel: Slope 1 dan -1")
    print("=" * 65)

    # Sistem dengan slope 1 dan -1:
    # y = x + 2    →  -x + y = 2   → matriks: [[-1, 1], [a, -1]] ...
    # Persamaan linear: slope 1 → y = x + 2 → x - y = -2
    #                   slope -1 → y = -x + 4 → x + y = 4
    # → Sistem: [ 1  -1] [x]   [-2]
    #           [ 1   1] [y] = [ 4]
    # Solusi: x=1, y=3

    A_conv = np.array([[1.0, -1.0], [1.0, 1.0]])
    b_conv = np.array([-2.0, 4.0])

    # Sistem tidak dominan (slope ±1): contoh divergen
    # Contoh: 2 persamaan dengan kemiringan sangat dekat
    # x + y = 3  dan  x - y = 1
    # Solusi: x=2, y=1
    # Matriks: [[1, 1], [1, -1]]
    # Diagonal dominansi: |1| vs |1| → TIDAK dominan (sama)
    A_div = np.array([[1.0,  1.0],
                      [1.0, -1.0]])
    b_div = np.array([3.0, 1.0])

    x_exact_conv = np.linalg.solve(A_conv, b_conv)
    x_exact_div  = np.linalg.solve(A_div, b_div)

    print(f"\nSistem konvergen (slope 1 dan -1, diagonal dominan):")
    print(f"  x - y = -2")
    print(f"  x + y =  4")
    print(f"  Solusi eksak: x={x_exact_conv[0]:.4f}, y={x_exact_conv[1]:.4f}")

    print(f"\nSistem potensi divergen (matrix tidak dominan diagonal):")
    print(f"  x + y = 3")
    print(f"  x - y = 1")
    print(f"  Solusi eksak: x={x_exact_div[0]:.4f}, y={x_exact_div[1]:.4f}")

    # Cek dominansi diagonal
    for name, A in [("Sistem 1", A_conv), ("Sistem 2", A_div)]:
        print(f"\n  {name} – dominansi diagonal:")
        for i in range(2):
            d = abs(A[i, i])
            o = abs(A[i, 1-i])
            print(f"    Baris {i+1}: |{A[i,i]:.0f}| vs |{A[i,1-i]:.0f}| → {'✓' if d > o else '✗ TIDAK dominan'}")

    # Iterasi Gauss-Seidel
    path1 = gauss_seidel_2d(A_conv, b_conv, x0=[0.0, 0.0], max_iter=15)
    path2 = gauss_seidel_2d(A_div,  b_div,  x0=[0.0, 0.0], max_iter=15)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Soal 11.14 – Gauss-Seidel: Slope 1 dan -1", fontsize=14, fontweight='bold')

    for ax, A, b, path, x_exact, title in [
        (axes[0], A_conv, b_conv, path1, x_exact_conv,
         "Sistem Konvergen\n(x - y = -2 | x + y = 4)"),
        (axes[1], A_div,  b_div,  path2, x_exact_div,
         "Sistem Berosilasi\n(x + y = 3 | x - y = 1)")
    ]:
        xv = np.linspace(-2, 6, 200)

        # Plot garis persamaan
        # Persamaan 1: A[0,0]*x + A[0,1]*y = b[0]  → y = (b[0] - A[0,0]*x) / A[0,1]
        if A[0, 1] != 0:
            y1 = (b[0] - A[0, 0] * xv) / A[0, 1]
            ax.plot(xv, y1, 'b-', linewidth=2, label='Persamaan 1')

        if A[1, 1] != 0:
            y2 = (b[1] - A[1, 0] * xv) / A[1, 1]
            ax.plot(xv, y2, 'r-', linewidth=2, label='Persamaan 2')

        # Plot jalur iterasi
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, 'g--o', markersize=5, linewidth=1.5, label='Jalur GS', zorder=5)

        for k, (xi, yi) in enumerate(zip(px[:6], py[:6])):
            ax.annotate(f'k={k}', (xi, yi), textcoords='offset points',
                        xytext=(5, 5), fontsize=8)

        ax.plot(*x_exact, 'k*', markersize=15, label=f'Solusi eksak\n({x_exact[0]:.2f}, {x_exact[1]:.2f})')

        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('prob_11_14_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_14_plot.png")
    plt.show()

    print("\n--- Kesimpulan ---")
    print("  Gauss-Seidel TIDAK menjamin konvergensi jika matriks")
    print("  tidak memenuhi syarat dominansi diagonal.")
    print("  Untuk slope 1 dan -1 (sistem ortogonal), jika |a_ii| ≤ Σ|a_ij|,")
    print("  iterasi dapat berosilasi atau gagal konvergen.")


if __name__ == "__main__":
    main()
