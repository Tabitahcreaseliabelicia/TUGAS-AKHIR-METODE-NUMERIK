"""
Soal 11.26 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Kembangkan program Gauss-Seidel yang user-friendly dalam Python.
Uji program dengan menduplikasi hasil Contoh 11.3.

Contoh 11.3 (dari buku) adalah sistem 3×3:
    [ 0.8  -0.4   0  ] [x1]   [ 41  ]
    [-0.4   0.8  -0.4] [x2] = [ 25  ]
    [ 0    -0.4   0.8] [x3]   [105  ]

Fitur program:
  - Input matriks sembarang atau mode demo
  - Penyusunan ulang baris otomatis (pivot-like dominansi diagonal)
  - Relaxation opsional
  - Tampilkan tabel iterasi
  - Plot konvergensi
"""

import numpy as np
import matplotlib.pyplot as plt


def check_diagonal_dominance(A):
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            return False
    return True


def rearrange_for_dominance(A, b):
    """Coba susun ulang baris agar dominan diagonal."""
    n = A.shape[0]
    A_new = A.copy()
    b_new = b.copy()
    for col in range(n):
        # Cari baris dengan elemen terbesar di kolom ini
        max_row = col + np.argmax(np.abs(A_new[col:, col]))
        if max_row != col:
            A_new[[col, max_row]] = A_new[[max_row, col]]
            b_new[[col, max_row]] = b_new[[max_row, col]]
    return A_new, b_new


def gauss_seidel_solver(A, b, x0=None, tol=5.0, max_iter=100, lam=1.0, verbose=True):
    """
    Gauss-Seidel solver dengan tampilan iterasi lengkap.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []

    if verbose:
        header = f"{'Iter':<6}" + "".join(f"{'x'+str(i+1):>14}" for i in range(n)) + f"{'εs_max':>12}"
        print(f"\n  {header}")
        print("  " + "-" * (6 + 14 * n + 12))

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]

        eps_vals = [abs((x[i] - x_old[i]) / x[i]) * 100 for i in range(n) if abs(x[i]) > 1e-14]
        max_eps = max(eps_vals) if eps_vals else 0.0
        history.append({'iter': iteration, 'x': x.copy(), 'eps': max_eps})

        if verbose:
            row = f"  {iteration:<6}" + "".join(f"{xi:>14.6f}" for xi in x) + f"{max_eps:>12.4f}%"
            print(row)

        if max_eps < tol:
            break

    return x, history


def run_example_11_3():
    """Duplikasi Contoh 11.3 dari buku."""
    print("\n  [MODE DEMO: Contoh 11.3 (sistem tridiagonal dari 11.1)]")
    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ], dtype=float)
    b = np.array([41.0, 25.0, 105.0])
    return A, b


def get_system_input():
    while True:
        try:
            n = int(input("  Masukkan ukuran sistem (n): "))
            if n < 2:
                print("  n harus ≥ 2.")
                continue
            break
        except ValueError:
            print("  Input tidak valid.")

    A = np.zeros((n, n))
    print(f"\n  Masukkan matriks A ({n}×{n}):")
    for i in range(n):
        for j in range(n):
            while True:
                try:
                    A[i, j] = float(input(f"    A[{i+1},{j+1}] = "))
                    break
                except ValueError:
                    print("    Input tidak valid.")

    b = np.zeros(n)
    print(f"\n  Masukkan vektor b:")
    for i in range(n):
        while True:
            try:
                b[i] = float(input(f"    b[{i+1}] = "))
                break
            except ValueError:
                print("    Input tidak valid.")

    return A, b


def main():
    print("=" * 65)
    print("  SOAL 11.26 – PROGRAM GAUSS-SEIDEL (USER-FRIENDLY)")
    print("=" * 65)
    print("\n  Menyelesaikan Ax = b menggunakan metode iteratif Gauss-Seidel.")
    print("  Menduplikasi hasil Contoh 11.3 dari buku.")

    print("\n  Pilih mode:")
    print("  1. Demo otomatis (Contoh 11.3 dari buku)")
    print("  2. Input manual")
    choice = input("\n  Pilihan Anda [1/2]: ").strip()

    if choice == '2':
        A, b = get_system_input()
    else:
        A, b = run_example_11_3()

    # Toleransi dan relaxation
    try:
        tol = float(input(f"\n  Toleransi εs (%) [default=5.0]: ") or "5.0")
    except ValueError:
        tol = 5.0

    try:
        lam = float(input(f"  Faktor relaxation λ [default=1.0]: ") or "1.0")
    except ValueError:
        lam = 1.0

    try:
        max_iter = int(input(f"  Iterasi maksimum [default=100]: ") or "100")
    except ValueError:
        max_iter = 100

    print(f"\n--- Informasi Sistem ---")
    print(f"  n = {len(b)}, εs = {tol}%, λ = {lam}, max_iter = {max_iter}")

    # Cek dominansi diagonal
    dom = check_diagonal_dominance(A)
    print(f"\n  Dominansi diagonal: {'✓ Ya' if dom else '✗ Tidak → susun ulang baris'}")

    if not dom:
        A, b = rearrange_for_dominance(A, b)
        dom2 = check_diagonal_dominance(A)
        print(f"  Setelah susun ulang: {'✓ Berhasil dominan' if dom2 else '✗ Masih tidak dominan (mungkin divergen)'}")

    # Jalankan Gauss-Seidel
    x, history = gauss_seidel_solver(A, b, tol=tol, lam=lam, max_iter=max_iter, verbose=True)

    print(f"\n--- SOLUSI (setelah {len(history)} iterasi) ---")
    for i, xi in enumerate(x, 1):
        print(f"  x{i} = {xi:.8f}")

    # Verifikasi
    resid = np.linalg.norm(A @ x - b)
    x_ref = np.linalg.solve(A, b)
    print(f"\n--- VERIFIKASI ---")
    print(f"  Norma residu ||Ax - b|| = {resid:.2e}")
    print(f"  Solusi eksak (numpy):    {np.round(x_ref, 6)}")

    # Plot konvergensi
    iters = [h['iter'] for h in history]
    eps_vals = [h['eps'] for h in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(iters, eps_vals, 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=tol, color='r', linestyle='--', label=f'εs = {tol}%')
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Error Relatif Maks (%)')
    ax.set_title(f'Soal 11.26 – Konvergensi Gauss-Seidel (λ={lam})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('prob_11_26_plot.png', dpi=150, bbox_inches='tight')
    print("\nPlot konvergensi disimpan: prob_11_26_plot.png")
    plt.show()

    print("\n  Program selesai. Terima kasih!")


if __name__ == "__main__":
    main()
