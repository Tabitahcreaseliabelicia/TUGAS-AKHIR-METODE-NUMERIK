"""
Soal 11.17 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Diberikan sistem persamaan nonlinear simultan:

    f(x, y) = 4 - y - 2x²
    g(x, y) = 8 - y² - 4x

(a) Tentukan dua pasang nilai (x, y) yang memenuhi persamaan ini
    (gunakan solver numerik / fsolve)
(b) Tentukan tebakan awal mana (x = -6 to 6, y = -6 to 6)
    yang menghasilkan masing-masing solusi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def system(vars):
    x, y = vars
    f = 4 - y - 2 * x**2
    g = 8 - y**2 - 4 * x
    return [f, g]


def find_all_solutions():
    """Temukan semua solusi dengan berbagai tebakan awal."""
    solutions = []
    tol = 1e-8
    for x0 in np.linspace(-6, 6, 25):
        for y0 in np.linspace(-6, 6, 25):
            try:
                sol = fsolve(system, [x0, y0], full_output=True)
                x, info, ier, msg = sol
                if ier == 1:  # Konvergen
                    res = np.linalg.norm(system(x))
                    if res < 1e-8:
                        # Cek apakah solusi sudah ada
                        is_new = True
                        for s in solutions:
                            if np.linalg.norm(np.array(x) - np.array(s)) < 1e-4:
                                is_new = False
                                break
                        if is_new:
                            solutions.append(x.tolist())
            except:
                pass
    return solutions


def main():
    print("=" * 65)
    print("SOAL 11.17 – Sistem Persamaan Nonlinear Simultan")
    print("=" * 65)

    print("\nSistem persamaan:")
    print("  f(x, y) = 4 - y - 2x² = 0")
    print("  g(x, y) = 8 - y² - 4x = 0")

    # (a) Temukan semua solusi
    solutions = find_all_solutions()
    print(f"\n(a) Solusi yang ditemukan: {len(solutions)} pasang")
    for i, (x, y) in enumerate(solutions, 1):
        print(f"  Solusi {i}: x = {x:.6f},  y = {y:.6f}")
        print(f"             f(x,y) = {4 - y - 2*x**2:.2e},  g(x,y) = {8 - y**2 - 4*x:.2e}")

    # Plot kontur
    xv = np.linspace(-4, 4, 400)
    yv = np.linspace(-4, 4, 400)
    X, Y = np.meshgrid(xv, yv)

    F = 4 - Y - 2 * X**2
    G = 8 - Y**2 - 4 * X

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Soal 11.17 – Persamaan Nonlinear Simultan", fontsize=14, fontweight='bold')

    # Plot 1: Kontur f=0 dan g=0
    ax = axes[0]
    cf = ax.contour(X, Y, F, levels=[0], colors='blue', linewidths=2)
    cg = ax.contour(X, Y, G, levels=[0], colors='red', linewidths=2)

    for i, (x, y) in enumerate(solutions):
        ax.plot(x, y, 'ko', markersize=12, zorder=5)
        ax.annotate(f'({x:.2f}, {y:.2f})', (x, y),
                    textcoords='offset points', xytext=(8, 8), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    h1, _ = cf.legend_elements()
    h2, _ = cg.legend_elements()
    ax.legend([h1[0], h2[0]], ['f(x,y) = 0', 'g(x,y) = 0'], fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kurva f(x,y)=0 dan g(x,y)=0')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Plot 2: Peta konvergensi tebakan awal
    ax2 = axes[1]
    nx, ny = 30, 30
    x_grid = np.linspace(-6, 6, nx)
    y_grid = np.linspace(-6, 6, ny)
    sol_map = np.zeros((ny, nx))

    for i, x0 in enumerate(x_grid):
        for j, y0 in enumerate(y_grid):
            try:
                s = fsolve(system, [x0, y0], full_output=True)
                sol = s[0]
                ier = s[2]
                res = np.linalg.norm(system(sol))
                if ier == 1 and res < 1e-6:
                    for k, (xs, ys) in enumerate(solutions):
                        if abs(sol[0]-xs) < 0.1 and abs(sol[1]-ys) < 0.1:
                            sol_map[j, i] = k + 1
                            break
            except:
                pass

    im = ax2.pcolormesh(x_grid, y_grid, sol_map, cmap='Set2', shading='auto')
    plt.colorbar(im, ax=ax2, label='Nomor Solusi')
    for i, (x, y) in enumerate(solutions):
        ax2.plot(x, y, 'k*', markersize=15)
        ax2.annotate(f'Sol {i+1}\n({x:.2f},{y:.2f})', (x, y),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax2.set_xlabel('x₀ (tebakan awal)')
    ax2.set_ylabel('y₀ (tebakan awal)')
    ax2.set_title('(b) Peta Konvergensi Tebakan Awal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prob_11_17_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_17_plot.png")
    plt.show()

    print("\n(b) Interpretasi peta konvergensi:")
    print("  Warna berbeda menunjukkan tebakan awal yang berkonvergensi")
    print("  ke solusi yang berbeda. Batas antar warna = 'fractal boundary'.")


if __name__ == "__main__":
    main()
