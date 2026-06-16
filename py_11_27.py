"""
Soal 11.27 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Persamaan diferensial berikut menggambarkan kesetimbangan massa
untuk bahan kimia dalam kanal satu dimensi (steady-state):

    0 = D·(d²c/dx²) - U·(dc/dx) - k·c

di mana:
  c  = konsentrasi (g/m³)
  D  = koefisien difusi  = 2   m²/s
  U  = kecepatan aliran  = 1   m/s
  k  = laju peluruhan    = 0.02 s⁻¹

Kondisi batas:
  c(0)  = 80  g/m³
  c(10) = 20  g/m³

Diskritisasi beda hingga dengan Δx = 2 (domain 0–10 m, 5 simpul interior):
  D/Δx²·(c_{i-1} - 2c_i + c_{i+1}) - U/(2Δx)·(c_{i+1} - c_{i-1}) - k·c_i = 0

Kumpulkan koefisien → sistem linear Ac = b.
"""

import numpy as np
import matplotlib.pyplot as plt


def build_system(D, U, k, c0, cL, x_start, x_end, dx):
    """
    Bangun sistem linear Ac = b dari diskritisasi beda hingga
    persamaan diferensial: D·c'' - U·c' - k·c = 0
    """
    x = np.arange(x_start, x_end + dx/2, dx)
    n_total = len(x)
    n_inner = n_total - 2   # simpul interior

    # Koefisien stensil beda hingga
    alpha = D / dx**2 - U / (2 * dx)   # koefisien c_{i-1}
    beta  = -2 * D / dx**2 - k          # koefisien c_i
    gamma = D / dx**2 + U / (2 * dx)   # koefisien c_{i+1}

    print(f"  Koefisien stensil:")
    print(f"    α (c_{{i-1}}) = D/Δx² - U/(2Δx) = {alpha:.4f}")
    print(f"    β (c_{{i}})   = -2D/Δx² - k      = {beta:.4f}")
    print(f"    γ (c_{{i+1}}) = D/Δx² + U/(2Δx) = {gamma:.4f}")

    # Bangun matriks A dan vektor b untuk simpul interior
    A = np.zeros((n_inner, n_inner))
    b = np.zeros(n_inner)

    for i in range(n_inner):
        A[i, i] = beta
        if i > 0:
            A[i, i-1] = alpha
        if i < n_inner - 1:
            A[i, i+1] = gamma

    # Kondisi batas
    b[0]  -= alpha * c0   # c_0 diketahui
    b[-1] -= gamma * cL   # c_L diketahui

    return A, b, x, alpha, beta, gamma


def analytical_solution(D, U, k, c0, cL, x_end):
    """
    Solusi analitik ODE: D·c'' - U·c' - k·c = 0
    Persamaan karakteristik: D·r² - U·r - k = 0
    r = (U ± sqrt(U² + 4Dk)) / (2D)
    """
    discriminant = U**2 + 4 * D * k
    r1 = (U + np.sqrt(discriminant)) / (2 * D)
    r2 = (U - np.sqrt(discriminant)) / (2 * D)

    # Kondisi batas: c(0) = c0, c(L) = cL
    # c(x) = A·e^(r1·x) + B·e^(r2·x)
    # Sistem: [1, 1; e^(r1·L), e^(r2·L)] [A, B]^T = [c0, cL]
    M = np.array([[1, 1],
                  [np.exp(r1 * x_end), np.exp(r2 * x_end)]])
    rhs = np.array([c0, cL])
    AB = np.linalg.solve(M, rhs)

    def c_exact(x):
        return AB[0] * np.exp(r1 * x) + AB[1] * np.exp(r2 * x)

    return c_exact, r1, r2


def main():
    print("=" * 65)
    print("SOAL 11.27 – ODE Kanal 1D → Sistem Linear (Beda Hingga)")
    print("=" * 65)

    # Parameter
    D   = 2.0    # koefisien difusi (m²/s)
    U   = 1.0    # kecepatan aliran (m/s)
    k   = 0.02   # laju peluruhan (1/s)
    c0  = 80.0   # konsentrasi di x=0 (g/m³)
    cL  = 20.0   # konsentrasi di x=10 (g/m³)
    x0  = 0.0
    xL  = 10.0
    dx  = 2.0    # ukuran langkah

    print(f"\nParameter:")
    print(f"  D = {D}, U = {U}, k = {k}")
    print(f"  c(0) = {c0}, c(10) = {cL}")
    print(f"  Δx = {dx}")

    x = np.arange(x0, xL + dx/2, dx)
    print(f"\nSimpul x: {x}")
    print(f"Simpul interior: {x[1:-1]}")

    print("\n--- Pembangunan Sistem Linear ---")
    A, b, x_all, alpha, beta, gamma = build_system(D, U, k, c0, cL, x0, xL, dx)

    n_inner = len(b)
    print(f"\nMatriks A ({n_inner}×{n_inner}):")
    print(np.round(A, 4))
    print(f"\nVektor b:")
    print(np.round(b, 4))

    # Selesaikan sistem
    c_inner = np.linalg.solve(A, b)
    c_full  = np.concatenate([[c0], c_inner, [cL]])

    print(f"\n--- Solusi (Beda Hingga, Δx={dx}) ---")
    for xi, ci in zip(x_all, c_full):
        print(f"  c({xi:.1f}) = {ci:.6f} g/m³")

    # Solusi analitik
    c_exact_fn, r1, r2 = analytical_solution(D, U, k, c0, cL, xL)
    print(f"\n--- Solusi Analitik ---")
    print(f"  r1 = {r1:.6f}, r2 = {r2:.6f}")
    x_fine = np.linspace(x0, xL, 200)
    c_analytic = c_exact_fn(x_fine)

    print(f"\n  Perbandingan di simpul:")
    for xi, ci_fd in zip(x_all, c_full):
        ci_exact = c_exact_fn(xi)
        err = abs(ci_fd - ci_exact)
        print(f"  x={xi:.1f}: FD={ci_fd:.4f}, Analitik={ci_exact:.4f}, Error={err:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_fine, c_analytic, 'b-', linewidth=2.5, label='Solusi Analitik', zorder=3)
    ax.plot(x_all, c_full, 'ro--', markersize=10, linewidth=1.5,
            label=f'Beda Hingga (Δx={dx})', zorder=4)
    ax.scatter([x0, xL], [c0, cL], s=120, color='green', zorder=5,
               label='Kondisi Batas')
    ax.set_xlabel('Posisi x (m)', fontsize=12)
    ax.set_ylabel('Konsentrasi c (g/m³)', fontsize=12)
    ax.set_title('Soal 11.27 – Distribusi Konsentrasi Kanal 1D\n'
                 r'$0 = D\frac{d^2c}{dx^2} - U\frac{dc}{dx} - kc$', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x0, xL)
    plt.tight_layout()
    plt.savefig('prob_11_27_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot disimpan: prob_11_27_plot.png")
    plt.show()

    print("\n--- Kesimpulan ---")
    print("  ODE turunan kedua berhasil dikonversi menjadi sistem linear")
    print("  melalui diskritisasi beda hingga tengah (central difference).")
    print(f"  Dengan Δx={dx}, diperoleh {n_inner} persamaan untuk {n_inner} unknowns.")


if __name__ == "__main__":
    main()
