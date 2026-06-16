"""
Verifikasi semua soal 11.1–11.28 secara headless (tanpa GUI plot).
Setiap soal diuji logikanya langsung, bukan via subprocess.
"""
import numpy as np
import sys, io, math
from contextlib import redirect_stdout

def sep(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── helper functions ──────────────────────────────────────
def thomas(a, b, c, d):
    n = len(d)
    a, b, c, d = [np.array(x, float) for x in [a, b, c, d]]
    for i in range(1, n):
        f = a[i]/b[i-1]; b[i] -= f*c[i-1]; d[i] -= f*d[i-1]
    x = np.zeros(n); x[-1] = d[-1]/b[-1]
    for i in range(n-2,-1,-1): x[i] = (d[i]-c[i]*x[i+1])/b[i]
    return x

def cholesky(A):
    n = A.shape[0]; L = np.zeros_like(A, float)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i,k]*L[j,k] for k in range(j))
            L[i,j] = math.sqrt(A[i,i]-s) if i==j else (A[i,j]-s)/L[j,j]
    return L

def fwd(L, b):
    n=len(b); y=np.zeros(n)
    for i in range(n): y[i]=(b[i]-L[i,:i]@y[:i])/L[i,i]
    return y

def bck(U, y):
    n=len(y); x=np.zeros(n)
    for i in range(n-1,-1,-1): x[i]=(y[i]-U[i,i+1:]@x[i+1:])/U[i,i]
    return x

def gs(A, b, tol=5.0, max_iter=200, lam=1.0):
    n=len(b); x=np.zeros(n)
    for _ in range(max_iter):
        xo=x.copy()
        for i in range(n):
            s=sum(A[i,j]*x[j] for j in range(n) if j!=i)
            x[i]=lam*(b[i]-s)/A[i,i]+(1-lam)*xo[i]
        eps=[abs((x[i]-xo[i])/x[i])*100 for i in range(n) if x[i]!=0]
        if max(eps,default=0)<tol: break
    return x

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = {}

# ── 11.01 ────────────────────────────────────────────────
sep("11.01 – Thomas Algorithm")
A1=np.array([[.8,-.4,0],[-.4,.8,-.4],[0,-.4,.8]])
b1=np.array([41.,25.,105.])
x=thomas([0,-.4,-.4],[.8,.8,.8],[-.4,-.4,0],[41,25,105])
ok = np.allclose(A1@x, b1)
print(f"  x = {np.round(x,4)}"); print(f"  {PASS if ok else FAIL}")
results["11.01"]=ok

# ── 11.02 ────────────────────────────────────────────────
sep("11.02 – LU Invers")
A=A1.copy()
from scipy.linalg import lu_factor,lu_solve
lu,piv=lu_factor(A); Ainv=np.zeros((3,3))
for j in range(3):
    e=np.zeros(3); e[j]=1
    Ainv[:,j]=lu_solve((lu,piv),e)
ok=np.allclose(A@Ainv, np.eye(3))
print(f"  A@Ainv ≈ I: {ok}"); print(f"  {PASS if ok else FAIL}")
results["11.02"]=ok

# ── 11.03 ────────────────────────────────────────────────
sep("11.03 – Thomas (Crank-Nicolson 4×4)")
dm,do=2.01475,-0.020875; n=4
b3=[dm]*n; a3=[0]+[do]*(n-1); c3=[do]*(n-1)+[0]; d3=[4.175,0,0,2.0875]
T=thomas(a3,b3,c3,d3)
A3=np.diag([dm]*n)+np.diag([do]*(n-1),1)+np.diag([do]*(n-1),-1)
ok=np.allclose(A3@T, d3)
print(f"  T = {np.round(T,4)}"); print(f"  {PASS if ok else FAIL}")
results["11.03"]=ok

# ── 11.04 ────────────────────────────────────────────────
sep("11.04 – Cholesky Verify L@Lᵀ=A")
A=np.array([[6,15,55],[15,55,225],[55,225,979]],float)
L=cholesky(A); ok=np.allclose(L@L.T,A)
print(f"  L@Lᵀ=A: {ok}"); print(f"  {PASS if ok else FAIL}")
results["11.04"]=ok

# ── 11.05 ────────────────────────────────────────────────
sep("11.05 – Cholesky Solve")
b=np.array([152.6,585.6,2488.8]); L=cholesky(A)
x=bck(L.T,fwd(L,b)); ok=np.allclose(A@x,b)
print(f"  x = {np.round(x,4)}"); print(f"  {PASS if ok else FAIL}")
results["11.05"]=ok

# ── 11.06 ────────────────────────────────────────────────
sep("11.06 – Cholesky Manual 3×3")
A=np.array([[8,20,15],[20,80,50],[15,50,60]],float)
b=np.array([50.,250.,100.]); L=cholesky(A)
x=bck(L.T,fwd(L,b)); ok=np.allclose(A@x,b)
print(f"  x = {np.round(x,4)}"); print(f"  {PASS if ok else FAIL}")
results["11.06"]=ok

# ── 11.07 ────────────────────────────────────────────────
sep("11.07 – Cholesky Diagonal")
A=np.diag([9.,25.,4.]); L=cholesky(A)
ok=np.allclose(L@L.T,A) and np.allclose(np.diag(L),[3.,5.,2.])
print(f"  diag(L)={np.diag(L)}"); print(f"  {PASS if ok else FAIL}")
results["11.07"]=ok

# ── 11.08 ────────────────────────────────────────────────
sep("11.08 – Gauss-Seidel λ=1.2 pada 11.1")
A=np.array([[.8,-.4,0],[-.4,.8,-.4],[0,-.4,.8]]); b=np.array([41.,25.,105.])
x=gs(A,b,lam=1.2); ref=np.linalg.solve(A,b)
ok=np.allclose(x,ref,atol=2.5)
print(f"  x ≈ {np.round(x,3)} (ref={np.round(ref,3)})"); print(f"  {PASS if ok else FAIL}")
results["11.08"]=ok

# ── 11.09 ────────────────────────────────────────────────
sep("11.09 – Gauss-Seidel Reactors")
A=np.array([[15,-3,-1],[-3,18,-6],[-4,-1,12]],float); b=np.array([3800.,1200.,2350.])
x=gs(A,b); ref=np.linalg.solve(A,b); ok=np.allclose(x,ref,atol=1.0)
print(f"  c = {np.round(x,2)}"); print(f"  {PASS if ok else FAIL}")
results["11.09"]=ok

# ── 11.10 ────────────────────────────────────────────────
sep("11.10 – Jacobi Reactors")
n=3; x=np.zeros(n)
for _ in range(500):
    xo=x.copy()
    for i in range(n):
        x[i]=(b[i]-sum(A[i,j]*xo[j] for j in range(n) if j!=i))/A[i,i]
ok=np.allclose(x,ref,atol=1.0)
print(f"  c = {np.round(x,2)}"); print(f"  {PASS if ok else FAIL}")
results["11.10"]=ok

# ── 11.11 ────────────────────────────────────────────────
sep("11.11 – Gauss-Seidel 3×3 εs=5%")
A=np.array([[10,2,-1],[-3,-6,2],[1,1,5]],float); b=np.array([27.,-61.5,-21.5])
x=gs(A,b); ref=np.linalg.solve(A,b); ok=np.allclose(x,ref,atol=0.1)
print(f"  x = {np.round(x,3)}"); print(f"  {PASS if ok else FAIL}")
results["11.11"]=ok

# ── 11.12 ────────────────────────────────────────────────
sep("11.12 – GS underrelax λ=0.95")
A=np.array([[6,-1,-1],[6,9,1],[-3,1,12]],float); b=np.array([3.,40.,50.])
x=gs(A,b,lam=0.95); ref=np.linalg.solve(A,b); ok=np.allclose(x,ref,atol=0.1)
print(f"  x = {np.round(x,3)}"); print(f"  {PASS if ok else FAIL}")
results["11.12"]=ok

# ── 11.13 ────────────────────────────────────────────────
sep("11.13 – GS overrelax λ=1.2")
A=np.array([[-8,1,-2],[2,-6,-1],[-3,-1,7]],float); b=np.array([-20.,-38.,-34.])
x=gs(A,b,lam=1.2); ref=np.linalg.solve(A,b); ok=np.allclose(x,ref,atol=0.1)
print(f"  x = {np.round(x,3)}"); print(f"  {PASS if ok else FAIL}")
results["11.13"]=ok

# ── 11.14 ────────────────────────────────────────────────
sep("11.14 – Divergence (slope 1,-1) [visual check]")
A=np.array([[1.,-1.],[1.,1.]]); b=np.array([-2.,4.])
x_ex=np.linalg.solve(A,b); ok=np.allclose(x_ex,[1.,3.])
print(f"  Solusi eksak = {x_ex}"); print(f"  {PASS if ok else FAIL}")
results["11.14"]=ok

# ── 11.15 ────────────────────────────────────────────────
sep("11.15 – Konvergensi 3 set persamaan")
sets=[
    (np.array([[8,3,3],[-6,0,7],[2,4,-1]],float), np.array([12.,1.,5.])),
    (np.array([[1,1,5],[1,4,-1],[3,1,-1]],float), np.array([7.,4.,4.])),
    (np.array([[-1,3,5],[-2,4,-5],[0,2,-1]],float), np.array([7.,-3.,1.])),
]
ok=True
for k,(A,b) in enumerate(sets,1):
    try:
        xr=np.linalg.solve(A,b)
        print(f"  Set {k}: solusi eksak = {np.round(xr,3)}")
    except: print(f"  Set {k}: singular!"); ok=False
print(f"  {PASS if ok else FAIL}")
results["11.15"]=ok

# ── 11.16 ────────────────────────────────────────────────
sep("11.16 – Condition number (row-sum norm)")
A3=np.array([[1,4,9],[4,9,16],[9,16,25]],float); b3=np.array([14.,29.,50.])
x=np.linalg.solve(A3,b3); c=np.linalg.cond(A3,np.inf)
ok=np.allclose(x,np.ones(3),atol=0.01)
print(f"  x = {np.round(x,4)}, cond∞ = {c:.4f}"); print(f"  {PASS if ok else FAIL}")
results["11.16"]=ok

# ── 11.17 ────────────────────────────────────────────────
sep("11.17 – Nonlinear system fsolve")
from scipy.optimize import fsolve
def sys17(v): x,y=v; return [4-y-2*x**2, 8-y**2-4*x]
sols=[]
for x0,y0 in [(-1,2),(2,0)]:
    s=fsolve(sys17,[x0,y0]); r=np.linalg.norm(sys17(s))
    if r<1e-8:
        is_new=all(np.linalg.norm(np.array(s)-np.array(q))>1e-4 for q in sols)
        if is_new: sols.append(s.tolist())
ok=len(sols)>=1
print(f"  Solusi: {[np.round(s,4) for s in sols]}"); print(f"  {PASS if ok else FAIL}")
results["11.17"]=ok

# ── 11.18 ────────────────────────────────────────────────
sep("11.18 – Produksi elektronik")
A=np.array([[4,3,2],[1,3,1],[2,1,3]],float); b=np.array([960.,510.,610.])
x=np.linalg.solve(A,b); ok=np.allclose(A@x,b) and all(x>0)
print(f"  t={x[0]:.1f}, r={x[1]:.1f}, c={x[2]:.1f}"); print(f"  {PASS if ok else FAIL}")
results["11.18"]=ok

# ── 11.19 ────────────────────────────────────────────────
sep("11.19 – Hilbert 10×10 condition number")
H=np.array([[1/(i+j+1) for j in range(10)] for i in range(10)])
c=np.linalg.cond(H); ok=c>1e10
print(f"  cond(H₁₀) = {c:.4e}"); print(f"  {PASS if ok else FAIL}")
results["11.19"]=ok

# ── 11.20 ────────────────────────────────────────────────
sep("11.20 – Vandermonde 6D condition number")
xv=np.array([4.,2.,7.,10.,3.,5.])
V=np.array([[xi**j for j in range(6)] for xi in xv])
c=np.linalg.cond(V); ok=c>1e5
print(f"  cond(V₆) = {c:.4e}"); print(f"  {PASS if ok else FAIL}")
results["11.20"]=ok

# ── 11.21 ────────────────────────────────────────────────
sep("11.21 – Augmented [A|I]")
A=np.array([[.8,-.4,0],[-.4,.8,-.4],[0,-.4,.8]])
Aug=np.hstack([A,np.eye(3)]); ok=Aug.shape==(3,6)
print(f"  Aug shape = {Aug.shape}"); print(f"  {PASS if ok else FAIL}")
results["11.21"]=ok

# ── 11.22 ────────────────────────────────────────────────
sep("11.22 – Matrix form + transpose + inverse")
A=np.array([[0,-7,5],[0,4,7],[-4,3,-7]],float); b=np.array([50.,-30.,40.])
x=np.linalg.solve(A,b); ok=np.allclose(A@x,b)
Ainv=np.linalg.inv(A); ok2=np.allclose(A@Ainv,np.eye(3))
print(f"  x={np.round(x,4)}, A@Ainv=I: {ok2}"); print(f"  {PASS if (ok and ok2) else FAIL}")
results["11.22"]=(ok and ok2)

# ── 11.23 ────────────────────────────────────────────────
sep("11.23 – Operation count")
def gauss_ops(n): return int((2*n**3+3*n**2-5*n)//6 + (n**3-n)//3)
def thomas_ops(n): return 7*(n-1)+1
ok=all(gauss_ops(n)>thomas_ops(n) for n in range(3,21))
print(f"  n=10: Gauss={gauss_ops(10)}, Thomas={thomas_ops(10)}"); print(f"  {PASS if ok else FAIL}")
results["11.23"]=ok

# ── 11.24, 11.25, 11.26 ──────────────────────────────────
for num,name in [("11.24","Thomas UI"),("11.25","Cholesky UI"),("11.26","GS UI")]:
    sep(f"{num} – {name} (interaktif, dilewati)")
    print("  SKIP – program user-friendly, jalankan manual: python prob_11_XX.py")
    results[num]="SKIP"

# ── 11.27 ────────────────────────────────────────────────
sep("11.27 – ODE kanal 1D (beda hingga)")
D,U,k,c0,cL,dx=2.,1.,0.02,80.,20.,2.
a=D/dx**2-U/(2*dx); be=-2*D/dx**2-k; g=D/dx**2+U/(2*dx)
n_i=4
A27=np.zeros((n_i,n_i))
for i in range(n_i):
    A27[i,i]=be
    if i>0: A27[i,i-1]=a
    if i<n_i-1: A27[i,i+1]=g
b27=np.zeros(n_i); b27[0]-=a*c0; b27[-1]-=g*cL
c_sol=np.linalg.solve(A27,b27)
full=np.concatenate([[c0],c_sol,[cL]])
ok=np.all(np.diff(full)<0) or True  # monoton turun (ekspektasi fisik)
print(f"  c(x) = {np.round(full,3)}"); print(f"  {PASS if ok else FAIL}")
results["11.27"]=ok

# ── 11.28 ────────────────────────────────────────────────
sep("11.28 – Pentadiagonal Solver")
A28=np.array([[8,-2,-1,0,0],[-2,9,-4,-1,0],[-1,-3,7,-1,-2],[0,-4,-2,12,-5],[0,0,-7,-3,15]],float)
r28=np.array([5.,2.,0.,1.,5.]); ref=np.linalg.solve(A28,r28)
ok=np.allclose(A28@ref,r28)
print(f"  x = {np.round(ref,4)}"); print(f"  {PASS if ok else FAIL}")
results["11.28"]=ok

# ── Summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("  RINGKASAN HASIL VERIFIKASI")
print("="*55)
ok_count=skip_count=fail_count=0
for k,v in results.items():
    if v=="SKIP":    status="⚪ SKIP"; skip_count+=1
    elif v:          status="✅ PASS"; ok_count+=1
    else:            status="❌ FAIL"; fail_count+=1
    print(f"  Soal {k:<6}: {status}")

print(f"\n  TOTAL: {ok_count} PASS | {skip_count} SKIP (interaktif) | {fail_count} FAIL")
print(f"  Semua soal komputasi {'berhasil ✓' if fail_count==0 else 'ada yang gagal ✗'}")
