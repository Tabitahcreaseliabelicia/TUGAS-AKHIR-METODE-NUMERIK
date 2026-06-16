"""
Soal 11.15 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Dari tiga set persamaan linear berikut, identifikasi set mana yang
TIDAK DAPAT diselesaikan menggunakan metode iteratif seperti Gauss-Seidel.
Tunjukkan menggunakan jumlah iterasi apapun bahwa solusinya tidak konvergen.
Nyatakan kriteria konvergensi Anda.

Set One:
  8x + 3y + 3z = 12
 -6x + 7z = 1
  2x + 4y - z = 5

Set Two:
  x + y + 5z = 7
  x + 4y - z = 4
  3x + y - z = 4

Set Three:
 -x + 3y + 5z = 7
 -2x + 4y - 5z = -3
  2y - z = 1
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=1e-4, max_iter=50, lam=1.0):
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            if A[i, i] == 0:
                return x, history, False  # Pivot nol, tidak bisa iterasi
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]
        eps = [abs((x[i] - x_old[i]) / x[i]) * 100 for i in range(n) if abs(x[i]) > 1e-12]
        max_eps = max(eps) if eps else 0
        history.append({'iter': iteration, 'x': x.copy(), 'eps': max_eps, 'diverged': max_eps > 1e6})
        if max_eps < tol:
            return x, history, True  # Konvergen
        if max_eps > 1e8:
            return x, history, False  # Divergen
    return x, history, False  # Tidak konvergen dalam batas iterasi


def check_dominance(A, name):
    """Periksa dominansi diagonal dan spectral radius."""
    n = A.shape[0]
    print(f"\n  {name} – Dominansi Diagonal:")
    dominant = True
    for i in range(n):
        diag = abs(A[i, i])
        off  = sum(abs(A[i, j]) for j in range(n) if j != i)
        is_dom = diag > off
        if not is_dom:
            dominant = False
        print(f"    Baris {i+1}: |{diag:.1f}| vs {off:.1f} → {'✓ DOMINAN' if is_dom else '✗ tidak dominan'}")

    # Spectral radius dari matriks iterasi Jacobi
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    try:
        M = -np.linalg.inv(D) @ L_plus_U
        eigenvals = np.linalg.eigvals(M)
        sr = max(abs(eigenvals))
        print(f"    Spectral radius (Jacobi) ρ = {sr:.4f} → {'✓ Konvergen' if sr < 1 else '✗ Divergen'}")
    except:
        print("    Tidak dapat menghitung spectral radius.")
        sr = None

    return dominant, sr


def main():
    print("=" * 65)
    print("SOAL 11.15 – Analisis Konvergensi Gauss-Seidel (3 Set Persamaan)")
    print("=" * 65)

    sets = {
        "Set One": {
            'A': np.array([[ 8,  3,  3],
                           [-6,  0,  7],
                           [ 2,  4, -1]], dtype=float),
            'b': np.array([12.0, 1.0, 5.0]),
            'eqs': ["8x + 3y + 3z = 12", "-6x + 7z = 1", "2x + 4y - z = 5"]
        },
        "Set Two": {
            'A': np.array([[ 1,  1,  5],
                           [ 1,  4, -1],
                           [ 3,  1, -1]], dtype=float),
            'b': np.array([7.0, 4.0, 4.0]),
            'eqs': ["x + y + 5z = 7", "x + 4y - z = 4", "3x + y - z = 4"]
        },
        "Set Three": {
            'A': np.array([[-1,  3,  5],
                           [-2,  4, -5],
                           [ 0,  2, -1]], dtype=float),
            'b': np.array([7.0, -3.0, 1.0]),
            'eqs': ["-x + 3y + 5z = 7", "-2x + 4y - 5z = -3", "2y - z = 1"]
        }
    }

    results = {}

    for name, data in sets.items():
        A = data['A']
        b = data['b']
        print(f"\n{'='*50}")
        print(f"  {name}")
        for eq in data['eqs']:
            print(f"    {eq}")

        dominant, sr = check_dominance(A, name)

        # Coba iterasi Gauss-Seidel
        x, history, converged = gauss_seidel(A, b, max_iter=50, tol=1e-4)

        print(f"\n  Hasil iterasi Gauss-Seidel (maks 10 iterasi ditampilkan):")
        print(f"  {'Iter':<6} {'x':>10} {'y':>10} {'z':>10} {'εs_max':>12}")
        for h in history[:10]:
            print(f"  {h['iter']:<6} {h['x'][0]:>10.4f} {h['x'][1]:>10.4f} {h['x'][2]:>10.4f} {h['eps']:>12.4f}%")
        if len(history) > 10:
            print(f"  ... ({len(history)} total iterasi)")

        status = "✓ KONVERGEN" if converged else "✗ TIDAK KONVERGEN (divergen/stuck)"
        print(f"\n  Status: {status} dalam {len(history)} iterasi")

        # Solusi eksak
        try:
            x_exact = np.linalg.solve(A, b)
            print(f"  Solusi eksak: x={x_exact[0]:.4f}, y={x_exact[1]:.4f}, z={x_exact[2]:.4f}")
        except np.linalg.LinAlgError:
            print("  Sistem singular, tidak ada solusi unik.")

        results[name] = converged

    print("\n" + "="*65)
    print("KESIMPULAN:")
    for name, conv in results.items():
        print(f"  {name}: {'✓ Konvergen' if conv else '✗ TIDAK konvergen dengan Gauss-Seidel'}")

    print("\nKriteria konvergensi Gauss-Seidel:")
    print("  1. Dominansi diagonal ketat: |a_ii| > Σ|a_ij| untuk semua i")
    print("  2. Spectral radius matriks iterasi ρ < 1")
    print("  Jika salah satu tidak terpenuhi, konvergensi tidak dijamin.")


if __name__ == "__main__":
    main()
