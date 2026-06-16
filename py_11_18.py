"""
Soal 11.18 – Numerical Methods for Engineers (Chapra & Canale)
==============================================================
Sebuah perusahaan elektronik memproduksi transistor, resistor, dan
computer chips. Kebutuhan bahan baku (copper, zinc, glass) per unit:

              Copper  Zinc  Glass
  Transistor:   4      1     2
  Resistor:     3      3     1
  Comp. chip:   2      1     3

Satu minggu tersedia: 960 unit copper, 510 unit zinc, 610 unit glass.
Tentukan jumlah transistor (t), resistor (r), dan chip (c) yang
harus diproduksi minggu ini.

Sistem persamaan:
  4t + 3r + 2c = 960   (copper)
   t + 3r +  c = 510   (zinc)
  2t +  r + 3c = 610   (glass)
"""

import numpy as np


def main():
    print("=" * 60)
    print("SOAL 11.18 – Masalah Produksi Elektronik (Sistem Linear 3×3)")
    print("=" * 60)

    print("\nTabel kebutuhan bahan baku:")
    print("  ┌─────────────┬────────┬──────┬───────┐")
    print("  │  Komponen   │ Copper │ Zinc │ Glass │")
    print("  ├─────────────┼────────┼──────┼───────┤")
    print("  │ Transistor  │   4    │   1  │   2   │")
    print("  │ Resistor    │   3    │   3  │   1   │")
    print("  │ Comp. chip  │   2    │   1  │   3   │")
    print("  ├─────────────┼────────┼──────┼───────┤")
    print("  │ Tersedia    │  960   │  510 │  610  │")
    print("  └─────────────┴────────┴──────┴───────┘")

    # Matriks koefisien [copper; zinc; glass] per [transistor, resistor, chip]
    A = np.array([
        [4, 3, 2],   # copper
        [1, 3, 1],   # zinc
        [2, 1, 3]    # glass
    ], dtype=float)

    b = np.array([960.0, 510.0, 610.0])

    print("\nSistem persamaan:")
    print("  4t + 3r + 2c = 960  (copper)")
    print("   t + 3r +  c = 510  (zinc)")
    print("  2t +  r + 3c = 610  (glass)")

    # Selesaikan menggunakan numpy
    x = np.linalg.solve(A, b)
    t, r, c = x

    print(f"\n--- Solusi ---")
    print(f"  Transistor (t) = {t:.2f} unit")
    print(f"  Resistor   (r) = {r:.2f} unit")
    print(f"  Chip       (c) = {c:.2f} unit")

    # Verifikasi
    print("\n--- Verifikasi penggunaan bahan baku ---")
    materials = ['Copper', 'Zinc', 'Glass']
    for i, (mat, avail) in enumerate(zip(materials, b)):
        used = A[i] @ x
        print(f"  {mat}: digunakan = {used:.2f}, tersedia = {avail:.0f} ✓")

    # Invers dan transpose
    A_T   = A.T
    A_inv = np.linalg.inv(A)

    print("\n--- Transpose Matriks A ---")
    print(A_T)

    print("\n--- Invers Matriks A ---")
    print(np.round(A_inv, 6))

    # Condition number
    cond = np.linalg.cond(A)
    print(f"\n--- Condition Number ---")
    print(f"  cond(A) = {cond:.4f}")
    print("  (Nilai mendekati 1 = well-conditioned)")


if __name__ == "__main__":
    main()
