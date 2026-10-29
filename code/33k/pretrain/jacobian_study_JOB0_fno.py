# =====================================================================================
#  CHANGE ONLY THIS.  0 -> trains FNO,  1 -> trains ResNet.
JOB = 0
# =====================================================================================
"""Jacobian-quality study on four chaotic 1D PDEs.

For every checkpoint of every trained emulator we compare the true autoregressive error
against two first-order predictors that share the SAME measured defect d_n:

    e_n                                    true rollout error of the model
    e^f_{n+1} = J_f(u_n) e^f_n + d_n       propagated by the MODEL's Jacobian
    e^P_{n+1} = J_P(u_n) e^P_n + d_n       propagated by the SOLVER's Jacobian

    d_n = f(u_n) - P(u_n),  evaluated on the TRUE trajectory, identical in both.
    Both start from e_0 = 0.  The only difference is which Jacobian propagates.

The gap between the two measures how far the learned Jacobian has drifted from the true
one -- a quantity the one-step training loss never sees.

4 PDEs x 1 config x 5 seeds = 20 runs per notebook.  Checkpoints every 30k updates.
Outputs: one 2-panel figure per checkpoint, plus curves.npz / summary.json / a zip.
"""
import os, sys, json, time, math, glob, subprocess, importlib.util

_miss = [m for m in ("equinox", "optax") if not importlib.util.find_spec(m)]
if _miss:
    from importlib.metadata import version, PackageNotFoundError
    _pin = []
    for _p in ("jax", "jaxlib"):
        try: _pin.append(f"{_p}=={version(_p)}")
        except PackageNotFoundError: pass
    print(f"[setup] pip install {_miss} (pinning {_pin})", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_pin, *_miss], check=True)
    importlib.invalidate_caches()

import jax
jax.config.update("jax_enable_x64", True)     # the stiff ETDRK4 contour needs float64
import numpy as np, jax.numpy as jnp, equinox as eqx, optax
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT   = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out")
BUDGET_H = float(os.environ.get("BUDGET_H", 0.1 if SMOKE else 10.0))
ARCH  = ["fno", "resnet"][JOB]
for d in ("figs", "ckpt", "curves"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
T0 = time.time()
hrs = lambda: (time.time() - T0) / 3600.0
el  = lambda: (time.time() - T0) / 60.0
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s, flush=True)

# ------------------------------------------------------------------ PDE configurations
NPTS = 128 if SMOKE else 256
CFG = [
    dict(tag="ks_L60",     fam="ks",   L=60.0,  dte=0.25, nsub=1, nch=1,
         ref="Edson+2019 arXiv:1902.09651"),
    dict(tag="gks_d0.2",   fam="gks",  L=200.0, delta=0.2, nu=1.0, dte=0.25, nsub=1, nch=1,
         ref="Gomes+2015 arXiv:1502.05266"),
    dict(tag="cgle_phaseL1", fam="cgle", L=250.0, b1=3.5, b3=1.4, dte=0.2, nsub=4, nch=2,
         ref="Chate 1993 arXiv:patt-sol/9310001"),
    dict(tag="nik_r0.02",  fam="nik",  L=200.0, r=0.02, dte=10.0, nsub=5, nch=1,
         ref="Matthews&Cox 2007 arXiv:0708.3735"),
]
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
TASKS = [(c, s) for s in SEEDS for c in CFG]
log(f"[job {JOB}] arch {ARCH} | {len(TASKS)} runs ({len(CFG)} PDEs x {len(SEEDS)} seeds) "
    f"| grid {NPTS} | budget {BUDGET_H:.1f}h")

# ------------------------------------------- ETDRK4 in JAX, float64, differentiable
# Written in JAX (not numpy) so that jax.jvp through it gives J_P exactly.  Full-circle
# contour: the Kassam-Trefethen half circle is only valid for real L, and gKS (dispersion)
# and CGLE (complex L) both break that.
def spectral(cfg):
    L, fam, N = cfg["L"], cfg["fam"], NPTS
    if fam == "cgle":
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        Lh = 1.0 - (1.0 + 1j * cfg["b1"]) * k ** 2
        b3 = cfg["b3"]
        def NL(v):
            A = jnp.fft.ifft(v)
            return jnp.fft.fft(-(b3 - 1j) * jnp.abs(A) ** 2 * A)
        return jnp.asarray(Lh, jnp.complex128), NL, True
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    if   fam == "ks":  Lh = k ** 2 - k ** 4
    elif fam == "gks": Lh = k ** 2 + 1j * cfg["delta"] * k ** 3 - cfg["nu"] * k ** 4
    elif fam == "nik": Lh = k ** 2 * (cfg["r"] - (1.0 - k ** 2) ** 2)
    else: raise ValueError(fam)
    mask = jnp.asarray((np.abs(k) < (2 / 3) * k.max()).astype(np.float64))   # 2/3 dealiasing
    kk = jnp.asarray(k)
    def NL(v):
        u = jnp.fft.irfft(v, n=N)
        return -0.5j * kk * jnp.fft.rfft(u ** 2) * mask
    return jnp.asarray(Lh, jnp.complex128), NL, False

def etdrk4_coeffs(Lh, h, M=64):
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)          # FULL circle
    LR = h * np.asarray(Lh)[:, None] + r[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Q  = h * np.mean((np.exp(LR / 2) - 1) / LR, 1)
        f1 = h * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1)
        f2 = h * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, 1)
        f3 = h * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, 1)
    out = [np.exp(h * np.asarray(Lh)), np.exp(h * np.asarray(Lh) / 2), Q, f1, f2, f3]
    return [jnp.asarray(np.where(np.isfinite(a), a, 0.0), jnp.complex128) for a in out]

def make_stepper(cfg):
    """Returns step_real: (nch, N) float64 -> (nch, N) float64, one emulator step."""
    Lh, NL, cx = spectral(cfg)
    h = cfg["dte"] / cfg["nsub"]
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(Lh, h)
    N = NPTS
    def to_spec(u):   # (nch,N) real -> spectral
        return jnp.fft.fft(u[0] + 1j * u[1]) if cx else jnp.fft.rfft(u[0])
    def to_real(v):
        if cx:
            A = jnp.fft.ifft(v); return jnp.stack([A.real, A.imag], 0)
        return jnp.fft.irfft(v, n=N)[None]
    def sub(v):
        Nv = NL(v);  a = E2 * v + Q * Nv
        Na = NL(a);  b = E2 * v + Q * Na
        Nb = NL(b);  c = E2 * a + Q * (2 * Nb - Nv)
        Nc = NL(c)
        return E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    def step_real(u):
        v = to_spec(u)
        v = jax.lax.fori_loop(0, cfg["nsub"], lambda _, z: sub(z), v)
        return to_real(v)
    return jax.jit(step_real)

def random_ic(cfg, rng, n, nm=6):
    x = np.linspace(0, cfg["L"], NPTS, endpoint=False)
    u = np.zeros((n, cfg["nch"], NPTS))
    for i in range(n):
        for ch in range(cfg["nch"]):
            for m in range(1, nm + 1):
                u[i, ch] += rng.normal() / m * np.sin(2*np.pi*m*x/cfg["L"] + rng.uniform(0, 2*np.pi))
    return 0.5 * u / (np.abs(u).max(axis=(1, 2), keepdims=True) + 1e-9)

def make_data(cfg, step, ntraj, nsteps, seed, warm):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(random_ic(cfg, rng, ntraj), jnp.float64)
    roll = jax.jit(lambda u0, m: jax.lax.scan(lambda s, _: (jax.vmap(step)(s),) * 2, u0, None, length=m),
               static_argnums=1)
    u, _ = roll(u, warm)
    _, tr = roll(u, nsteps)                                   # (nsteps, ntraj, nch, N)
    out = jnp.concatenate([u[None], tr], 0).swapaxes(0, 1)    # (ntraj, nsteps+1, nch, N)
    if not bool(jnp.isfinite(out).all()) or float(jnp.abs(out).max()) > 1e3:
        raise RuntimeError(f"{cfg['tag']}: solver unstable (max {float(jnp.abs(out).max()):.3g})")
    return np.asarray(out, np.float32)

# ------------------------------------------------------------------------ networks
class SpectralConv1d(eqx.Module):
    wr: jnp.ndarray; wi: jnp.ndarray; modes: int = eqx.field(static=True)
    def __init__(self, cin, cout, modes, key):
        sc = 1.0 / (cin * cout) ** 0.5; k1, k2 = jax.random.split(key)
        self.wr = jax.random.normal(k1, (cin, cout, modes), dtype=jnp.float32) * sc
        self.wi = jax.random.normal(k2, (cin, cout, modes), dtype=jnp.float32) * sc
        self.modes = modes
    def __call__(self, x):
        xh = jnp.fft.rfft(x, axis=-1)
        m = min(self.modes, xh.shape[-1])
        w = (self.wr + 1j * self.wi)[:, :, :m]
        o = jnp.einsum("in,ion->on", xh[:, :m], w)
        full = jnp.zeros((w.shape[1], xh.shape[-1]), dtype=o.dtype).at[:, :m].set(o)
        return jnp.fft.irfft(full, n=x.shape[-1], axis=-1)

class Conv1(eqx.Module):
    w: jnp.ndarray; b: jnp.ndarray; k: int = eqx.field(static=True)
    def __init__(self, cin, cout, k, key, zero=False):
        self.w = (jnp.zeros((cout, cin, k), jnp.float32) if zero else
                  jax.random.normal(key, (cout, cin, k), dtype=jnp.float32) / (cin * k) ** 0.5)
        self.b = jnp.zeros(cout, jnp.float32); self.k = k
    def __call__(self, x):
        p = self.k // 2
        xp = jnp.concatenate([x[:, -p:], x, x[:, :p]], -1)[None] if p else x[None]
        y = jax.lax.conv_general_dilated(xp, self.w, (1,), "VALID",
                                         dimension_numbers=("NCW", "OIW", "NCW"))[0]
        return y + self.b[:, None]

class FNO(eqx.Module):                       # APEBench preset FNO;12;18;4;gelu ~ 32.5k params
    lift: Conv1; spec: list; pw: list; p1: Conv1; p2: Conv1
    def __init__(self, nch, key, width=18, modes=12, nblk=4):
        ks = jax.random.split(key, 2 * nblk + 3)
        self.lift = Conv1(nch, width, 1, ks[0])
        self.spec = [SpectralConv1d(width, width, modes, ks[1 + i]) for i in range(nblk)]
        self.pw   = [Conv1(width, width, 1, ks[1 + nblk + i]) for i in range(nblk)]
        self.p1 = Conv1(width, 64, 1, ks[-2]); self.p2 = Conv1(64, nch, 1, ks[-1], zero=True)
    def __call__(self, u):
        x = self.lift(u)
        for sc, pc in zip(self.spec, self.pw): x = jax.nn.gelu(sc(x) + pc(x))
        return u + self.p2(jax.nn.gelu(self.p1(x)))

class ResNet(eqx.Module):                    # APEBench preset Res;26;8;relu ~ 32.9k params
    lift: Conv1; blk: list; proj: Conv1
    def __init__(self, nch, key, width=26, nblk=8):
        ks = jax.random.split(key, 2 * nblk + 2)
        self.lift = Conv1(nch, width, 3, ks[0])
        self.blk = [(Conv1(width, width, 3, ks[1 + 2*i]), Conv1(width, width, 3, ks[2 + 2*i]))
                    for i in range(nblk)]
        self.proj = Conv1(width, nch, 3, ks[-1], zero=True)
    def __call__(self, u):
        x = self.lift(u)
        for a, b in self.blk: x = x + b(jax.nn.relu(a(x)))
        return u + self.proj(x)

make_net = lambda nch, key: FNO(nch, key) if ARCH == "fno" else ResNet(nch, key)
nparams = lambda p: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(p, eqx.is_array)))

# --------------------------------------------------------------------- run parameters
def _E(n, d): return type(d)(os.environ.get(n, d))
NTRAIN, TRAIN_H = (8, 20)  if SMOKE else (_E("NTRAIN", 128), _E("TRAIN_H", 50))
NTEST,  TEST_H  = (4, 40)  if SMOKE else (_E("NTEST", 30),  _E("TEST_H", 200))
NVAL,   VAL_H   = (4, 20)  if SMOKE else (16, 100)
UPDATES   = 600 if SMOKE else _E("UPDATES", 150_000)
CKPT_EVERY= 200 if SMOKE else _E("CKPT_EVERY", 30_000)      # <- checkpoint cadence
BATCH     = 8   if SMOKE else _E("BATCH", 32)
LR, WARMUP, LR_MIN = 1e-3, (20 if SMOKE else 2000), 1e-5
EVAL_EVERY= 100 if SMOKE else 5000
WARM_SOLVER = 50 if SMOKE else 300
NSTAR_THR = 0.25
OPT = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-1.0))

# ------------------------------------------------------- validation (must all PASS)
def validate():
    ok = True
    for cfg in CFG:
        step = make_stepper(cfg)
        rng = np.random.default_rng(0)
        u = jnp.asarray(random_ic(cfg, rng, 2), jnp.float64)
        for _ in range(WARM_SOLVER): u = jax.vmap(step)(u)
        w = jax.random.normal(jax.random.PRNGKey(1), u.shape, dtype=jnp.float64)
        jw = jax.jvp(lambda z: jax.vmap(step)(z), (u,), (w,))[1]
        eps = 1e-6
        fd = (jax.vmap(step)(u + eps * w) - jax.vmap(step)(u - eps * w)) / (2 * eps)
        r_fd = float(jnp.linalg.norm(jw - fd) / jnp.linalg.norm(fd))
        jw2 = jax.jvp(lambda z: jax.vmap(step)(z), (u,), (3 * w,))[1]
        r_lin = float(jnp.linalg.norm(jw2 - 3 * jw) / jnp.linalg.norm(jw))
        fin = bool(jnp.isfinite(u).all()) and float(jnp.abs(u).max()) < 1e3
        # leading growth rate of a tiny perturbation under the TRUE solver
        p = 1e-9 * jax.random.normal(jax.random.PRNGKey(2), u.shape, dtype=jnp.float64)
        lg, uu = [], u
        for _ in range(200):
            p = jax.jvp(lambda z: jax.vmap(step)(z), (uu,), (p,))[1]
            uu = jax.vmap(step)(uu); lg.append(np.log(np.asarray(jnp.linalg.norm(p, axis=(-2,-1)))))
        lg = np.array(lg); lo, hi = int(0.2*len(lg)), int(0.8*len(lg))
        fits = [np.polyfit(np.arange(lo, hi), lg[lo:hi, i], 1)[0]
                for i in range(lg.shape[1]) if np.isfinite(lg[lo:hi, i]).all()]
        lam = float(np.median(fits)) if fits else float("nan")
        # the grid must resolve the band where Re L_k > 0, or the system is not chaotic at all
        kmax = 2 * np.pi * (NPTS // 2) / cfg["L"] * (1.0 if cfg["fam"] == "cgle" else 2 / 3)
        kneed = {"ks": 1.0, "gks": 1.0, "cgle": 1.0,
                 "nik": float(np.sqrt(1 + np.sqrt(cfg.get("r", 0))))}[cfg["fam"]]
        res_ok = kmax > 1.5 * kneed
        good = (r_fd < 1e-6) and (r_lin < 1e-10) and fin and np.isfinite(lam) and (lam > 0) and res_ok
        ok &= good
        log(f"  {'PASS' if good else 'FAIL'} {cfg['tag']:<14} tangent-vs-FD {r_fd:.1e} | "
            f"linear {r_lin:.1e} | bounded {fin} | lambda/step {lam:+.4f} | "
            f"k_max {kmax:.2f} vs k_unstable {kneed:.2f} {'ok' if res_ok else 'UNDER-RESOLVED'}")
        cfg["lam"] = lam
    log("ALL PASS" if ok else "VALIDATION FAILED -- results below are not trustworthy")
    return ok

log("\n[validate] solver, its tangent, and chaoticity")
OK_ALL = validate()

# ----------------------------------------------------------------------- training
def train(cfg, step, seed, train_np, val_np, rd, deadline):
    os.makedirs(rd, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    net = make_net(cfg["nch"], key)
    P0, static = eqx.partition(net, eqx.is_array)
    mk = lambda q: eqx.combine(q, static)
    tr = jnp.asarray(train_np); va = jnp.asarray(val_np)
    NT, T1 = tr.shape[0], tr.shape[1]

    @eqx.filter_jit
    def tstep(q, ost, x, y, lr):
        def L(z): return jnp.mean((jax.vmap(mk(z))(x) - y) ** 2)
        lv, g = eqx.filter_value_and_grad(L)(q)
        up, nost = OPT.update(g, ost)
        up = jax.tree_util.tree_map(lambda a: a * lr, up)
        return eqx.apply_updates(q, up), nost, lv
    @eqx.filter_jit
    def valerr(q):
        m = mk(q)
        pr = jax.lax.scan(lambda s, _: (jax.vmap(m)(s),) * 2, va[:, 0], None, length=VAL_H)[1]
        pr = jnp.swapaxes(pr, 0, 1)
        e = jnp.sqrt(jnp.sum((pr - va[:, 1:VAL_H+1])**2, (-2,-1)) / jnp.sum(va[:, 1:VAL_H+1]**2, (-2,-1)))
        return jnp.exp(jnp.mean(jnp.log(jnp.clip(e.mean(0), 1e-12, 1e3))))

    p, ost = P0, OPT.init(eqx.filter(P0, eqx.is_array))
    rng = np.random.default_rng(1000 + seed)
    best, saved, hist = float("inf"), [], []
    for it in range(1, UPDATES + 1):
        ti = rng.integers(0, NT, BATCH); si = rng.integers(0, T1 - 1, BATCH)
        x = tr[ti, si]; y = tr[ti, si + 1]
        lr = LR * (it / WARMUP if it <= WARMUP else
                   LR_MIN/LR + (1-LR_MIN/LR) * 0.5 * (1 + math.cos(math.pi * (it-WARMUP)/max(UPDATES-WARMUP,1))))
        p, ost, lv = tstep(p, ost, x, y, jnp.asarray(lr, jnp.float32))
        if it % CKPT_EVERY == 0:
            eqx.tree_serialise_leaves(f"{rd}/ckpt_{it}.eqx", mk(p)); saved.append(it)
        if it % EVAL_EVERY == 0 or it == UPDATES:
            v = float(valerr(p)); hist.append(dict(it=it, loss=float(lv), val=v))
            if v < best: best = v; eqx.tree_serialise_leaves(f"{rd}/ckpt_best.eqx", mk(p))
            log(f"    [{cfg['tag']}/{ARCH}/s{seed}] {it:>7} loss {float(lv):.3e} val {v:.5f} "
                f"best {best:.5f} {el():.1f}m")
            if hrs() > deadline: log(f"    deadline at {it}/{UPDATES}"); break
    if not os.path.exists(f"{rd}/ckpt_best.eqx"):
        eqx.tree_serialise_leaves(f"{rd}/ckpt_best.eqx", mk(p))
    json.dump(dict(hist=hist, saved=saved, best_val=best, updates=it), open(f"{rd}/hist.json", "w"))
    return static, saved

# --------------------------------------------------- the three curves per checkpoint
def analyse(cfg, step, static, path, test_np):
    """Returns ‖e‖, ‖e^f‖, ‖e^P‖ relative to ‖u_true‖, plus the two vector mismatches."""
    model = eqx.tree_deserialise_leaves(path, make_net(cfg["nch"], jax.random.PRNGKey(0)))
    U64 = jnp.asarray(test_np, jnp.float64)          # (B, S+1, nch, N)
    S = U64.shape[1] - 1
    f32 = lambda z: z.astype(jnp.float32)
    f_of = lambda z: model(f32(z)).astype(jnp.float64)       # model in f32, interface in f64

    @jax.jit
    def go(U):
        nrm = lambda a: jnp.sqrt(jnp.sum(a ** 2, axis=(-2, -1)))
        pred = jnp.swapaxes(jax.lax.scan(
            lambda s, _: (jax.vmap(f_of)(s),) * 2, U[:, 0], None, length=S)[1], 0, 1)
        e_true = pred - U[:, 1:]
        d = jax.vmap(jax.vmap(f_of))(U[:, :S]) - jax.vmap(jax.vmap(step))(U[:, :S])  # f(u)-P(u)
        def prop(J):
            def body(e, i):
                Je = jax.vmap(lambda b, v: jax.jvp(J, (b,), (v,))[1])(U[:, i], e)
                return (Je + d[:, i],) * 2
            return jnp.swapaxes(jax.lax.scan(body, jnp.zeros_like(U[:, 0]), jnp.arange(S))[1], 0, 1)
        ef, eP = prop(f_of), prop(step)
        rel = lambda a: nrm(a - e_true) / jnp.clip(nrm(e_true), 1e-30)
        return nrm(e_true), nrm(ef), nrm(eP), nrm(U[:, 1:]), rel(ef), rel(eP)

    et, ef, eP, un, rf, rP = [np.asarray(z) for z in go(U64)]
    g = lambda a: np.exp(np.nanmean(np.log(np.where(np.isfinite(a) & (a > 0), a, np.nan)), 0))
    nstar = lambda m: int(np.argmax(m > NSTAR_THR) + 1) if (m > NSTAR_THR).any() else S + 1
    mf, mP = g(rf), g(rP)
    return dict(true=g(et/un), f=g(ef/un), P=g(eP/un), mis_f=mf, mis_P=mP,
                Nf=nstar(mf), NP=nstar(mP), S=S)

C_TRUE, C_F, C_P = "#2a78d6", "#eb6834", "#1baf7a"
def figure(tag, R, path):
    S = R["S"]; n = np.arange(1, S + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax[0].plot(n, R["true"], c=C_TRUE, lw=2.2, label=r"true  $\|e_n\|$")
    ax[0].plot(n, R["f"], c=C_F, lw=1.8, ls=(0, (5, 2)), label=r"$J_f$ predictor")
    ax[0].plot(n, R["P"], c=C_P, lw=1.8, ls=(0, (1.5, 1.5)), label=r"$J_P$ predictor (same $d_n$)")
    ax[0].set(xlabel="rollout step $n$", ylabel="relative error", title=tag,
              ylim=(0, 1.15 * max(float(np.nanmax(R["true"])), 1e-9)))
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(n, R["mis_f"], c=C_F, lw=1.8, label=r"$J_f$")
    ax[1].plot(n, R["mis_P"], c=C_P, lw=1.8, label=r"$J_P$")
    ax[1].axhline(NSTAR_THR, c="#8a8f98", ls=":", lw=1)
    ax[1].annotate(f"{int(NSTAR_THR*100)}% threshold", (0.98, NSTAR_THR + 0.01),
                   xycoords=("axes fraction", "data"), fontsize=7, color="#55595f", ha="right")
    for v, c_, lb in ((R["NP"], C_P, "$N^*_P$"), (R["Nf"], C_F, "$N^*_f$")):
        if v <= S:
            ax[1].axvline(v, c=c_, ls=":", lw=1.2)
            ax[1].annotate(f"{lb}={v}", (v, 0.97), xycoords=("data", "axes fraction"),
                           fontsize=8, color=c_, rotation=90, ha="right", va="top")
    ax[1].set(xlabel="rollout step $n$", ylabel=r"$\|\hat e_n-e_n\|\,/\,\|e_n\|$",
              title="predictor mismatch (vector)", ylim=(0, 1.0),
              xlim=(0, min(1.6 * max(R["Nf"], R["NP"]), S)))
    ax[1].legend(fontsize=8, frameon=False)
    for a_ in ax:
        a_.grid(True, which="both", lw=0.4, color="#dfe2e6"); a_.set_axisbelow(True)
        for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------ main loop
RES, ARR, DATA = {}, {}, {}
for i, (cfg, seed) in enumerate(TASKS):
    share = BUDGET_H * (i + 1) / len(TASKS)
    if hrs() > BUDGET_H: log(f"\n[job] out of budget; {len(TASKS)-i} runs not done"); break
    tag = f"{cfg['tag']}__{ARCH}__s{seed}"
    log(f"\n{'='*94}\n[RUN {i+1}/{len(TASKS)}] {tag}  (deadline {share:.2f}h, now {hrs():.2f}h)\n{'='*94}")
    try:
        if cfg["tag"] not in DATA:
            step = make_stepper(cfg); t = time.time()
            DATA[cfg["tag"]] = (step,
                make_data(cfg, step, NTRAIN, TRAIN_H, 1000, WARM_SOLVER),
                make_data(cfg, step, NVAL,   VAL_H,   4242, WARM_SOLVER),
                make_data(cfg, step, NTEST,  TEST_H,  773,  WARM_SOLVER))
            log(f"  [data] {cfg['tag']} train{DATA[cfg['tag']][1].shape} test{DATA[cfg['tag']][3].shape}"
                f"  rms {np.sqrt((DATA[cfg['tag']][1]**2).mean()):.3f}  ({time.time()-t:.0f}s)")
        step, trn, val, tst = DATA[cfg["tag"]]
        rd = f"{OUT}/ckpt/{tag}"
        static, saved = train(cfg, step, seed, trn, val, rd, share)
        log(f"  [ckpt] {len(saved)} milestones {saved} + best")
        for ck in [f"{rd}/ckpt_{m}.eqx" for m in saved] + [f"{rd}/ckpt_best.eqx"]:
            name = os.path.basename(ck)[5:-4]
            lbl = f"{tag}__{'best' if name=='best' else (f'{int(name)//1000}k' if int(name)>=1000 else name)}"
            R = analyse(cfg, step, static, ck, tst)
            RES[lbl] = {k: R[k] for k in ("Nf", "NP", "S")}
            RES[lbl].update(pde=cfg["tag"], arch=ARCH, seed=seed,
                            ckpt=(None if name == "best" else int(name)),
                            e_end=float(R["true"][-1]), lam=cfg.get("lam"))
            for w in ("true", "f", "P", "mis_f", "mis_P"): ARR[f"{lbl}/{w}"] = R[w]
            figure(f"{cfg['tag']}  {ARCH}  seed {seed}  "
                   f"{'best' if name=='best' else f'{int(name):,} updates'}", R,
                   f"{OUT}/figs/{lbl}.png")
            log(f"    [fig] {lbl:<36} e({R['S']})={R['true'][-1]:.3f}  "
                f"N*_f={R['Nf']:>4}  N*_P={R['NP']:>4}  ratio={R['Nf']/max(R['NP'],1):.2f}")
    except Exception as e:
        import traceback; log(f"[RUN FAILED] {tag}: {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1200:])

# --------------------------------------------------------------------------- summary
np.savez_compressed(f"{OUT}/curves/job{JOB}_curves.npz", **ARR)
json.dump(RES, open(f"{OUT}/job{JOB}_results.json", "w"), indent=1, default=str)

L = [f"JOB {JOB}  arch {ARCH}  |  {len(RES)} checkpoints from {len(set(r['seed'] for r in RES.values()))} seeds"
     f"  |  wall {el():.1f} min  |  validation {'PASS' if OK_ALL else 'FAIL'}",
     f"params: grid {NPTS}  train {NTRAIN}x{TRAIN_H}  test {NTEST}x{TEST_H}  updates {UPDATES}"
     f"  ckpt every {CKPT_EVERY}  batch {BATCH}  lr {LR}",
     "=" * 94,
     f"{'checkpoint':<40}{'e(end)':>9}{'N*_f':>7}{'N*_P':>7}{'ratio':>8}"]
for k in sorted(RES, key=lambda k: (RES[k]['pde'], RES[k]['seed'], RES[k]['ckpt'] or 10**9)):
    r = RES[k]
    L.append(f"{k:<40}{r['e_end']:>9.3f}{r['Nf']:>7}{r['NP']:>7}{r['Nf']/max(r['NP'],1):>8.2f}")
L += ["", "=" * 94, "PER-PDE MEDIAN OVER SEEDS (last checkpoint of each run)", "=" * 94]
for pde in [c["tag"] for c in CFG]:
    v = [r for r in RES.values() if r["pde"] == pde and r["ckpt"] is None]
    if v:
        L.append(f"  {pde:<16} lam/step {v[0]['lam']:+.4f}   N*_f {np.median([r['Nf'] for r in v]):>6.0f}"
                 f"   N*_P {np.median([r['NP'] for r in v]):>6.0f}"
                 f"   ratio {np.median([r['Nf']/max(r['NP'],1) for r in v]):>6.2f}   n={len(v)}")
summary = "\n".join(L)
open(f"{OUT}/job{JOB}_summary.txt", "w").write(summary + "\n\n=== FULL LOG ===\n" + "\n".join(LOG))

import zipfile
with zipfile.ZipFile(f"{OUT}/job{JOB}_figures.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f"{OUT}/figs/*.png")): z.write(f, os.path.basename(f))
print("\n" + "=" * 94 + "\nPASTE EVERYTHING BELOW THIS LINE\n" + "=" * 94)
print(summary)
print(f"\n[files] {OUT}/job{JOB}_summary.txt  {OUT}/job{JOB}_results.json  "
      f"{OUT}/job{JOB}_figures.zip  {OUT}/figs/*.png  {OUT}/ckpt/*/  wall {el():.1f} min")
