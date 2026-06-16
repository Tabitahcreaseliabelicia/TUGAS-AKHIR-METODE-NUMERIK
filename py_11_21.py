"""
Soal 11.21 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Diberikan matriks persegi [A], tulis satu perintah Python (setara
perintah MATLAB satu baris) yang membuat matriks baru [Aug] yang
terdiri dari matriks asli [A] yang ditambahi matriks identitas [I].

Ekuivalen MATLAB: Aug = [A, eye(size(A))]

Contoh menggunakan matriks dari Contoh 11.1:
    A = [0.8  -0.4   0  ]
        [-0.4  0.8  -0.4]
        [0    -0.4   0.8]
"""

import numpy as np


def main():
    print("=" * 65)
    print("SOAL 11.21 – Matriks Augmented [A | I] (Satu Baris Perintah)")
    print("=" * 65)

    # Matriks A dari Contoh 11.1
    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])

    n = A.shape[0]

    print("\nMatriks A:")
    print(A)

    # =====================================================
    # SATU BARIS PERINTAH Python (setara MATLAB: [A eye(n)])
    Aug = np.hstack([A, np.eye(n)])
    # =====================================================

    print(f"\nPerintah satu baris Python:")
    print(f"  Aug = np.hstack([A, np.eye(n)])")
    print(f"\n  (MATLAB: Aug = [A, eye(size(A))])")

    print(f"\nMatriks Augmented [A | I] ({n}×{2*n}):")
    print(Aug)

    # Verifikasi: menyelesaikan [A|I] untuk mendapatkan A^{-1}
    print("\n--- Manfaat [A|I]: Basis Eliminasi Gauss untuk Invers ---")
    print("  Dengan mengaplikasikan eliminasi Gauss pada [A|I],")
    print("  bagian kanan menjadi A^{-1} setelah transformasi selesai.")

    # Demonstrasi: gunakan np.linalg.inv vs eliminasi manual
    A_inv_ref = np.linalg.inv(A)
    print(f"\n  A⁻¹ (numpy.linalg.inv):")
    print(np.round(A_inv_ref, 6))

    # Eliminasi Gauss-Jordan pada [A|I]
    Aug_work = Aug.copy()
    for col in range(n):
        # Normalisasi baris pivot
        pivot = Aug_work[col, col]
        Aug_work[col] /= pivot
        for row in range(n):
            if row != col:
                factor = Aug_work[row, col]
                Aug_work[row] -= factor * Aug_work[col]

    A_inv_gj = Aug_work[:, n:]
    print(f"\n  A⁻¹ (Gauss-Jordan dari [A|I]):")
    print(np.round(A_inv_gj, 6))

    match = np.allclose(A_inv_ref, A_inv_gj)
    print(f"\n  {'✓ Kedua metode menghasilkan invers yang sama!' if match else '✗ Ada perbedaan!'}")

    # Contoh array lain untuk demonstrasi satu baris
    print("\n--- Demonstrasi dengan matriks lain ---")
    B = np.array([[2, 1], [5, 3]], dtype=float)
    print(f"\n  B = {B.tolist()}")
    Aug_B = np.hstack([B, np.eye(B.shape[0])])  # satu baris!
    print(f"  np.hstack([B, np.eye(n)]) =")
    print(f"  {Aug_B}")


if __name__ == "__main__":
    main()
