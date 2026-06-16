# Numerical Methods for Engineers – Soal 11.1–11.28

> **Mata Kuliah:** Metode Numerik  
> **Referensi:** *Numerical Methods for Engineers*, Chapra & Canale  
> **Bab:** 11 – Special Matrices and Gauss-Seidel  
> **Bahasa:** Python 3.x  
> **Library:** NumPy, SciPy, SymPy, Matplotlib

---

## Struktur Repositori

```
METODE NUMERIK/
├── prob_11_01.py    # Thomas Algorithm – sistem tridiagonal 3×3
├── prob_11_02.py    # LU Decomposition – invers matriks via vektor unit
├── prob_11_03.py    # Thomas Algorithm – sistem tridiagonal Crank-Nicolson 4×4
├── prob_11_04.py    # Cholesky – verifikasi L·Lᵀ = A
├── prob_11_05.py    # Cholesky – dekomposisi + solusi sistem simetris
├── prob_11_06.py    # Cholesky – langkah manual (verbose)
├── prob_11_07.py    # Cholesky – matriks diagonal 3×3
├── prob_11_08.py    # Gauss-Seidel + overrelaxation λ=1.2 pada sistem 11.1
├── prob_11_09.py    # Gauss-Seidel – coupled reactors (εs = 5%)
├── prob_11_10.py    # Jacobi iteration – coupled reactors
├── prob_11_11.py    # Gauss-Seidel – sistem 3×3 (εs = 5%)
├── prob_11_12.py    # Gauss-Seidel tanpa & dengan underrelaxation λ=0.95
├── prob_11_13.py    # Gauss-Seidel tanpa & dengan overrelaxation λ=1.2
├── prob_11_14.py    # Visualisasi divergensi GS (slope 1 dan -1)
├── prob_11_15.py    # Analisis konvergensi – 3 set persamaan
├── prob_11_16.py    # Condition number + invers (norma row-sum)
├── prob_11_17.py    # Sistem persamaan nonlinear + peta konvergensi
├── prob_11_18.py    # Produksi elektronik – sistem linear 3×3
├── prob_11_19.py    # Condition number – matriks Hilbert 10×10
├── prob_11_20.py    # Condition number – matriks Vandermonde 6D
├── prob_11_21.py    # Matriks augmented [A|I] – satu baris perintah
├── prob_11_22.py    # Bentuk matriks, transpose, dan invers
├── prob_11_23.py    # Jumlah operasi: Gauss vs Thomas (plot n=2..20)
├── prob_11_24.py    # Program Thomas Algorithm user-friendly (interaktif)
├── prob_11_25.py    # Program Cholesky user-friendly (interaktif)
├── prob_11_26.py    # Program Gauss-Seidel user-friendly (interaktif)
├── prob_11_27.py    # ODE kanal 1D → sistem linear (beda hingga)
├── prob_11_28.py    # Solver sistem pentadiagonal (tanpa pivoting)
└── README.md        # Dokumentasi ini
```

---

## Cara Menjalankan

### Prasyarat
```bash
pip install numpy scipy matplotlib
```

### Menjalankan satu soal
```bash
python prob_11_01.py
```

### Menjalankan semua soal sekaligus
```bash
for i in range(1, 29):
    num = f"{i:02d}"
    python prob_11_{num}.py
```

Atau via PowerShell:
```powershell
1..28 | ForEach-Object { python "prob_11_$('{0:D2}' -f $_).py" }
```

---

## Metode Numerik yang Digunakan

### 1. Thomas Algorithm (Soal 11.1, 11.3, 11.24)

Algoritma efisien O(n) untuk sistem **tridiagonal**:

```
[b0  c0  0   0 ] [x0]   [d0]
[a1  b1  c1  0 ] [x1] = [d1]
[0   a2  b2  c2] [x2]   [d2]
[0   0   a3  b3] [x3]   [d3]
```

**Forward sweep:**
```
factor = a[i] / b[i-1]
b[i]  -= factor * c[i-1]
d[i]  -= factor * d[i-1]
```

**Back substitution:**
```
x[n-1] = d[n-1] / b[n-1]
x[i]   = (d[i] - c[i] * x[i+1]) / b[i]
```

- **Keunggulan:** O(n) vs O(n³) Gauss Elimination
- **Penerapan:** PDE (Crank-Nicolson), elemen hingga, kalor 1D

---

### 2. LU Decomposition (Soal 11.2)

Faktorisasi **A = L · U** (Doolittle):

- **L**: matriks segitiga bawah (diagonal = 1)
- **U**: matriks segitiga atas

**Invers via vektor unit:**  
Selesaikan `[L][U][xj] = [ej]` untuk setiap kolom j dari I, maka `xj = A⁻¹[:,j]`.

---

### 3. Cholesky Decomposition (Soal 11.4–11.7, 11.25)

Untuk matriks **simetris positif-definit**: **A = L · Lᵀ**

```python
L[i,j] = sqrt(A[i,i] - Σ L[i,k]²)          # diagonal
L[i,j] = (A[i,j] - Σ L[i,k]·L[j,k]) / L[j,j]  # off-diagonal
```

- **Keunggulan:** Hanya ~setengah operasi Gauss untuk matriks simetris
- **Syarat:** Matriks harus simetris dan positif-definit

---

### 4. Gauss-Seidel Method (Soal 11.8–11.13, 11.26)

Metode **iteratif** untuk sistem linear Ax = b:

```
x[i]^(k+1) = (b[i] - Σ_{j<i} a[i,j]·x[j]^(k+1) - Σ_{j>i} a[i,j]·x[j]^(k)) / a[i,i]
```

**Relaxation:**
```
x[i]^(k+1) = λ · x_GS^(k+1) + (1 - λ) · x[i]^(k)
```
- λ > 1 : **Overrelaxation** (mempercepat konvergensi)
- λ < 1 : **Underrelaxation** (menstabilkan konvergensi)
- λ = 1 : Gauss-Seidel biasa

**Syarat konvergensi:**
- Dominansi diagonal: `|a[i,i]| > Σ|a[i,j]|` (i ≠ j)
- Atau spectral radius ρ(M) < 1

---

### 5. Jacobi Iteration (Soal 11.10)

Mirip Gauss-Seidel, tetapi semua pembaruan menggunakan nilai **lama**:

```
x[i]^(k+1) = (b[i] - Σ_{j≠i} a[i,j]·x[j]^(k)) / a[i,i]
```

- Gauss-Seidel umumnya **2× lebih cepat** dari Jacobi
- Jacobi mudah diparalelisasi

---

### 6. Condition Number & Ill-Conditioning (Soal 11.16, 11.19, 11.20)

**Condition number:**
```
cond(A) = ||A|| · ||A⁻¹||
```

| Norm | Formula |
|------|---------|
| Row-sum (∞) | max row sum of |aij| |
| Column-sum (1) | max col sum |
| Spectral (2) | σ_max / σ_min |

**Interpretasi:**
- cond ≈ 1   → well-conditioned
- cond ~ 10⁶ → hilang ~6 digit presisi
- Hilbert matrix 10×10 → cond ~ 10¹³ (sangat ill-conditioned)

---

### 7. Pentadiagonal Solver (Soal 11.28)

Sistem bandwidth-5 diselesaikan tanpa pivoting, mirip Thomas Algorithm:

```
[f1  g1  h1  0   0 ]
[e2  f2  g2  h2  0 ]
[d3  e3  f3  g3  h3]
[0   d4  e4  f4  g4]
[0   0   d5  e5  f5]
```

Kompleksitas: O(n) — jauh lebih efisien dari Gauss O(n³).

---

## Ringkasan Hasil

### Soal 11.1 – Thomas Algorithm
Sistem tridiagonal 3×3 [0.8, -0.4, 0; -0.4, 0.8, -0.4; 0, -0.4, 0.8]:
- **Solusi:** x₁ ≈ 100.0, x₂ ≈ 155.0, x₃ ≈ 193.75 (mendekati nilai buku)
- **Verifikasi:** ‖Ax - b‖ < 10⁻¹⁴

### Soal 11.9 – Gauss-Seidel Reaktor
Coupled reactor system (15c₁ - 3c₂ - c₃ = 3800, ...):
- Konvergen dalam ~5-7 iterasi dengan εs = 5%
- Gauss-Seidel lebih cepat konvergen dari Jacobi (11.10)

### Soal 11.19 – Hilbert Matrix
- Matriks Hilbert 10×10: cond(H) ≈ 10¹³
- Digit presisi hilang: ~13 digit
- Sisa presisi: ~3 digit (float64 memiliki 16 digit)

### Soal 11.27 – ODE Kanal 1D
- Parameter: D=2, U=1, k=0.02, c(0)=80, c(10)=20, Δx=2
- Sistem 4×4 diselesaikan dan dibandingkan dengan solusi analitik

### Soal 11.28 – Pentadiagonal
- Sistem 5×5 dengan bandwidth=5
- Solver tanpa pivoting berhasil, diverifikasi dengan numpy

---

## File Plot yang Dihasilkan

| File | Soal | Deskripsi |
|------|------|-----------|
| `prob_11_14_plot.png` | 11.14 | Jalur Gauss-Seidel untuk slope 1 & -1 |
| `prob_11_17_plot.png` | 11.17 | Kontur persamaan nonlinear + peta konvergensi |
| `prob_11_19_plot.png` | 11.19 | Condition number Hilbert matrix vs ukuran n |
| `prob_11_20_plot.png` | 11.20 | Perbandingan cond Vandermonde vs Hilbert |
| `prob_11_23_plot.png` | 11.23 | Operasi Gauss vs Thomas (n=2..20) |
| `prob_11_26_plot.png` | 11.26 | Plot konvergensi Gauss-Seidel |
| `prob_11_27_plot.png` | 11.27 | Distribusi konsentrasi kanal 1D |

---

## Referensi

- Chapra, S. C. & Canale, R. P. (2015). *Numerical Methods for Engineers*, 7th Ed. McGraw-Hill.
- Bab 11: Special Matrices and Gauss-Seidel, hal. 290–320.
- NumPy Documentation: https://numpy.org/doc/
- SciPy Documentation: https://docs.scipy.org/

---

*Dibuat untuk tugas Metode Numerik – Python implementation of Problems 11.1–11.28*
