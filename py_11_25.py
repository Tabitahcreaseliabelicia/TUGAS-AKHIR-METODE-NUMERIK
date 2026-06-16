"""
Soal 11.25 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Kembangkan program Cholesky yang user-friendly dalam Python.
Uji program dengan menduplikasi hasil Contoh 11.2.

Fitur:
  - Input matriks simetris dari pengguna (atau demo otomatis)
  - Validasi: simetris & positif-definit
  - Tampilkan matriks L langkah demi langkah
  - Selesaikan sistem Ax = b
  - Verifikasi residu
"""

import numpy as np
import math


def cholesky_verbose(A, show_steps=True):
    """Dekomposisi Cholesky dengan tampilan langkah."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    if show_steps:
        print("\n  === Dekomposisi Cholesky: A = L × Lᵀ ===")

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError("Matriks tidak positif-definit!")
                L[i, j] = math.sqrt(val)
                if show_steps:
                    print(f"  L[{i+1},{j+1}] = sqrt({A[i,i]:.4f} - {s:.4f}) = {L[i,j]:.6f}")
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                if show_steps:
                    print(f"  L[{i+1},{j+1}] = ({A[i,j]:.4f} - {s:.4f}) / {L[j,j]:.4f} = {L[i,j]:.6f}")
    return L


def forward_sub(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def back_sub(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


def is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A):
    try:
        eigenvals = np.linalg.eigvalsh(A)
        return all(eigenvals > 0)
    except:
        return False


def get_matrix_input():
    """Input matriks simetris dari pengguna."""
    while True:
        try:
            n = int(input("  Masukkan ukuran matriks (n): "))
            if n < 2:
                print("  n harus ≥ 2.")
                continue
            break
        except ValueError:
            print("  Input tidak valid.")

    A = np.zeros((n, n))
    print(f"\n  Masukkan elemen matriks ({n}×{n}), baris per baris:")
    for i in range(n):
        for j in range(n):
            while True:
                try:
                    A[i, j] = float(input(f"    A[{i+1},{j+1}] = "))
                    break
                except ValueError:
                    print("    Input tidak valid.")

    b = np.zeros(n)
    print(f"\n  Masukkan vektor RHS b[1..{n}]:")
    for i in range(n):
        while True:
            try:
                b[i] = float(input(f"    b[{i+1}] = "))
                break
            except ValueError:
                print("    Input tidak valid.")
    return A, b


def run_example_11_2():
    """Duplikasi Contoh 11.2 dari buku."""
    print("\n  [MODE DEMO: Contoh 11.2]")
    A = np.array([
        [  6,  15,   55],
        [ 15,  55,  225],
        [ 55, 225,  979]
    ], dtype=float)
    b = np.array([152.6, 585.6, 2488.8])
    print("  Matriks A:")
    print(A)
    print("  Vektor b:", b)
    return A, b


def main():
    print("=" * 60)
    print("  SOAL 11.25 – PROGRAM CHOLESKY DECOMPOSITION (USER-FRIENDLY)")
    print("=" * 60)
    print("\n  Menyelesaikan sistem simetris positif-definit Ax = b")
    print("  menggunakan Cholesky Decomposition.")

    print("\n  Pilih mode:")
    print("  1. Demo otomatis (Contoh 11.2 dari buku)")
    print("  2. Input manual")

    choice = input("\n  Pilihan Anda [1/2]: ").strip()

    if choice == '2':
        A, b = get_matrix_input()
    else:
        A, b = run_example_11_2()

    # Validasi
    print("\n--- Validasi Matriks ---")
    sym = is_symmetric(A)
    pd  = is_positive_definite(A)
    print(f"  Simetris       : {'✓ Ya' if sym else '✗ Tidak!'}")
    print(f"  Positif-definit: {'✓ Ya' if pd else '✗ Tidak!'}")

    if not sym:
        print("\n  ⚠ Matriks tidak simetris. Coba simetrisasi: A = (A + Aᵀ)/2")
        A = (A + A.T) / 2
        print("  Matriks telah disimetrisasi.")

    if not pd:
        print("\n  ⚠ Matriks tidak positif-definit! Cholesky mungkin gagal.")

    try:
        L = cholesky_verbose(A, show_steps=True)

        print(f"\n--- Matriks L ---")
        print(np.round(L, 6))

        print(f"\n--- Verifikasi: L × Lᵀ ---")
        print(np.round(L @ L.T, 6))
        print(f"  {'✓ L × Lᵀ = A' if np.allclose(L @ L.T, A) else '✗ GAGAL'}")

        # Selesaikan sistem
        print(f"\n--- Substitusi Maju: L·y = b ---")
        y = forward_sub(L, b)
        print(f"  y = {np.round(y, 6)}")

        print(f"\n--- Substitusi Mundur: Lᵀ·x = y ---")
        x = back_sub(L.T, y)

        print(f"\n--- SOLUSI ---")
        for i, xi in enumerate(x, 1):
            print(f"  x{i} = {xi:.8f}")

        # Verifikasi
        resid = np.linalg.norm(A @ x - b)
        x_ref = np.linalg.solve(A, b)
        print(f"\n--- VERIFIKASI ---")
        print(f"  Norma residu ||Ax - b|| = {resid:.2e}")
        print(f"  Solusi referensi numpy:  {np.round(x_ref, 8)}")
        print(f"  {'✓ Solusi valid!' if resid < 1e-8 else '✗ Periksa kembali!'}")

    except ValueError as e:
        print(f"\n  ⚠ Error: {e}")

    print("\n  Program selesai. Terima kasih!")


if __name__ == "__main__":
    main()
