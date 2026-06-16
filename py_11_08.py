"""
Soal 11.8 – Numerical Methods for Engineers (Chapra & Canale)
=============================================================
Gunakan metode Gauss-Seidel untuk menyelesaikan sistem tridiagonal
dari Soal 11.1 (εs = 5%). Gunakan overrelaxation dengan λ = 1.2.

Sistem:
    [ 0.8  -0.4   0  ] [x1]   [ 41  ]
    [-0.4   0.8  -0.4] [x2] = [ 25  ]
    [ 0    -0.4   0.8] [x3]   [105  ]

Catatan: Sistem perlu diperiksa dominansi diagonal.
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=0.05, max_iter=100, lam=1.0):
    """
    Metode Gauss-Seidel dengan relaxation.

    Parameter:
        A       : matriks koefisien
        b       : vektor RHS
        x0      : tebakan awal (default: zeros)
        tol     : toleransi persen relatif (%)
        max_iter: iterasi maksimum
        lam     : faktor relaxation (1=tanpa, >1=over, <1=under)

    Return:
        x       : solusi
        history : riwayat per iterasi
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            # Hitung nilai Gauss-Seidel baru
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            # Terapkan relaxation
            x[i] = lam * x_new + (1 - lam) * x_old[i]

        # Hitung error persen relatif maks
        eps = []
        for i in range(n):
            if x[i] != 0:
                eps.append(abs((x[i] - x_old[i]) / x[i]) * 100)
        max_eps = max(eps) if eps else 0

        history.append({
            'iter': iteration,
            'x': x.copy(),
            'eps': max_eps
        })

        if max_eps < tol:
            break

    return x, history


def main():
    print("=" * 65)
    print("SOAL 11.8 – Gauss-Seidel + Overrelaxation (λ=1.2) pada Sistem 11.1")
    print("=" * 65)

    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])
    b = np.array([41.0, 25.0, 105.0])

    # Cek dominansi diagonal
    print("\nPemeriksaan dominansi diagonal:")
    for i in range(len(b)):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(len(b)) if j != i)
        status = "✓" if diag >= off else "✗"
        print(f"  Baris {i+1}: |{A[i,i]}| vs {off:.1f} → {status}")

    print("\nCatatan: Sistem tidak sepenuhnya dominan diagonal,")
    print("         konvergensi mungkin lambat atau memerlukan relaksasi.")

    lam = 1.2
    tol = 5.0  # 5%

    print(f"\n--- Gauss-Seidel TANPA relaxation (λ=1.0) ---")
    x1, hist1 = gauss_seidel(A, b, tol=tol, lam=1.0)
    print(f"{'Iter':<6} {'x1':>12} {'x2':>12} {'x3':>12} {'εs_max':>10}")
    for h in hist1:
        print(f"{h['iter']:<6} {h['x'][0]:>12.6f} {h['x'][1]:>12.6f} {h['x'][2]:>12.6f} {h['eps']:>10.4f}%")

    print(f"\n--- Gauss-Seidel DENGAN Overrelaxation (λ={lam}) ---")
    x2, hist2 = gauss_seidel(A, b, tol=tol, lam=lam)
    print(f"{'Iter':<6} {'x1':>12} {'x2':>12} {'x3':>12} {'εs_max':>10}")
    for h in hist2:
        print(f"{h['iter']:<6} {h['x'][0]:>12.6f} {h['x'][1]:>12.6f} {h['x'][2]:>12.6f} {h['eps']:>10.4f}%")

    # Solusi eksak (Thomas / numpy)
    x_exact = np.linalg.solve(A, b)
    print("\n--- Solusi Eksak (numpy) ---")
    for i, xi in enumerate(x_exact, 1):
        print(f"  x{i} = {xi:.6f}")

    print(f"\nKonvergen dalam {len(hist1)} iterasi (tanpa relaxation)")
    print(f"Konvergen dalam {len(hist2)} iterasi (dengan λ={lam})")


if __name__ == "__main__":
    main()
