"""Benchmark: C river-showdown kernel vs NumPy per-board loop, then extrapolate."""
import ctypes, os, time, sys
import numpy as np
sys.path.insert(0, "src")
from itertools import combinations
from pokertrainer.cards import parse_cards
from pokertrainer.ranges import expand_range
from pokertrainer.presets import BB_SRP, BTN_SRP
from pokertrainer.evaluator import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(HERE, "kernel.so")
if not os.path.exists(lib_path):
    raise SystemExit(
        f"Missing {lib_path}. Build it first:\n"
        f"  cc -O3 -shared -fPIC -o {lib_path} {os.path.join(HERE, 'kernel.c')}"
    )
lib = ctypes.CDLL(lib_path)
lib.river_pass.argtypes = [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_float)]*2 + \
    [ctypes.POINTER(ctypes.c_double)]*2 + [ctypes.c_double]*3 + [ctypes.POINTER(ctypes.c_double)]*2

flop = parse_cards("As7h2d")
oop = [c for c,_ in expand_range(BB_SRP, flop)]
ip  = [c for c,_ in expand_range(BTN_SRP, flop)]
def sub(l,n):
    idx=np.linspace(0,len(l)-1,min(n,len(l))).astype(int); return [l[i] for i in idx]
N = 160
oop, ip = sub(oop,N), sub(ip,N)
no, ni = len(oop), len(ip)
oc = np.array(oop); ic = np.array(ip)
print(f"combos {no}x{ni}", flush=True)

# compat B
B = np.ones((no,ni), np.float32)
for i in range(no):
    a,b = oc[i]
    B[i, (ic[:,0]==a)|(ic[:,1]==a)|(ic[:,0]==b)|(ic[:,1]==b)] = 0.0

# All river boards (flop + 2 runout cards)
used=set(flop); deck=[c for c in range(52) if c not in used]
boards = list(combinations(deck,2))
nb = len(boards)
print(f"river boards: {nb}", flush=True)

# Precompute E_all [nb,no,ni] float32 (win matrix per board, compat baked in)
t=time.time()
E_all = np.empty((nb,no,ni), np.float32)
for bi,(t1,t2) in enumerate(boards):
    b5=[*flop,t1,t2]
    ro=np.array([evaluate((a,b,*b5)) for a,b in oc])
    ri=np.array([evaluate((a,b,*b5)) for a,b in ic])
    gt = ro[:,None]>ri[None,:]
    E_all[bi] = B*np.where(gt,1.0,np.where(ro[:,None]==ri[None,:],0.5,0.0))
print(f"E_all build: {time.time()-t:.1f}s, mem {E_all.nbytes/1e6:.0f}MB", flush=True)

ri_v = (np.ones(ni)/ni).astype(np.float64)
ro_v = (np.ones(no)/no).astype(np.float64)
pot, eo, ei = 20.0, 7.0, 7.0

# --- NumPy per-board loop (mimics multistreet.py showdown work) ---
def numpy_pass():
    uo=np.zeros(no); ui=np.zeros(ni)
    for bi in range(nb):
        E=E_all[bi]
        uo += pot*(E@ri_v) - eo*(B@ri_v)
        ui += pot*((B-E).T@ro_v) - ei*(B.T@ro_v)
    return uo,ui

# --- C kernel ---
E_flat = np.ascontiguousarray(E_all).ravel()
B_flat = np.ascontiguousarray(B).ravel()
uo_c=np.zeros(no); ui_c=np.zeros(ni)
def c_pass():
    lib.river_pass(nb,no,ni,
        E_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ri_v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ro_v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pot,eo,ei,
        uo_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ui_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return uo_c,ui_c

# correctness
un, uni = numpy_pass(); uc, uci = c_pass()
print(f"max |numpy-C| diff OOP: {np.abs(un-uc).max():.2e}  IP: {np.abs(uni-uci).max():.2e}", flush=True)

# timing
K=10
t=time.time()
for _ in range(K): numpy_pass()
tp=(time.time()-t)/K
t=time.time()
for _ in range(K): c_pass()
tc=(time.time()-t)/K
print(f"\nNumPy showdown pass: {tp*1000:8.1f} ms/iter", flush=True)
print(f"C     showdown pass: {tc*1000:8.1f} ms/iter", flush=True)
print(f"speedup: {tp/tc:.0f}x", flush=True)

print("\n=== Extrapolation to a full river solve (1 board) ===", flush=True)
for iters in (500,1000):
    print(f"  {iters} iters:  NumPy ~{tp*iters:6.0f}s   C ~{tc*iters:6.1f}s", flush=True)
print(f"  12-board library @1000 iters:  NumPy ~{tp*1000*12/3600:.1f} h   C ~{tc*1000*12/60:.1f} min", flush=True)
print("  (showdown-only; real solve adds regret/tree work — small vs showdown)", flush=True)
