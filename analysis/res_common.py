"""Evidence for the four contributions: figures and LaTeX table bodies (33k + small models + noise runs)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import comb
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_all as LA

RES = LA.RESULTS
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")); os.makedirs(OUT, exist_ok=True)
ROOTS33 = list(LA.ROOTS)
R33, _ = LA.load()
LA.ROOTS = [os.path.join(RES, "small")]; RS, _ = LA.load()
LA.ROOTS = [os.path.join(RES, "33k_noise")]; RN, _ = LA.load()

PN = {"ks_L60": "KS", "gks_d0.2": "gKS", "cgle_phaseL1": "CGLE", "nik_r0.02": "Nik."}
AN = {"conv_s": "CNN", "resnet_s": "ResNet", "unet_s": "U-Net", "fno_s": "FNO", "fno32_s": "FNO", "ptf_s": "Transformer", "resnet": "ResNet", "fno": "FNO"}
AK = {"conv_s": "conv", "resnet_s": "resnet", "unet_s": "unet", "fno_s": "fno", "fno32_s": "fno", "ptf_s": "ptf", "resnet": "resnet", "fno": "fno"}
MK = {"conv": "s", "resnet": "o", "unet": "^", "fno": "D", "ptf": "P"}
PC = {"ks_L60": "#2a78d6", "gks_d0.2": "#eb6834", "cgle_phaseL1": "#1baf7a", "nik_r0.02": "#8a5cc2"}
AC = {"own100": "#1baf7a", "rand100": "#eb6834", "noise1": "#8a5cc2", "start": "#0b0b0b"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
TW = 6.27  # text width of the arXiv template (A4, 1in margins), inches
EXCL = {("small", "ks_L60", "conv_s")}


def style(ax):
    ax.grid(True, which="major", lw=0.4, color="#e4e3df"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gm(v):
    v = np.asarray(v, float); v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.mean(np.log(v)))) if v.size else np.nan


def n1(b):
    return int(round(1 / (b["vpt_cap"] / 200)))


def sign_p(w, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(max(w, n - w), n + 1)) / 2 ** n) if n else 1.0


def fm(x, d=2):
    if x is None or not np.isfinite(x):
        return "--"
    if abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0):
        e = int(np.floor(np.log10(abs(x)))); return f"${x / 10 ** e:.1f}\\!\\times\\!10^{{{e}}}$"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.{d}g}"


def write(name, lines):
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ per-seed records
CFG33 = [("ks_L60", "resnet", range(5)), ("nik_r0.02", "resnet", range(5)), ("cgle_phaseL1", "resnet", range(5)), ("gks_d0.2", "resnet", [0, 1]),
         ("ks_L60", "fno", range(5)), ("gks_d0.2", "fno", range(5)), ("cgle_phaseL1", "fno", range(5))]
ARS = {"ks_L60": ["conv_s", "resnet_s", "unet_s", "fno_s", "ptf_s"], "gks_d0.2": ["conv_s", "resnet_s", "unet_s", "fno32_s", "ptf_s"],
       "cgle_phaseL1": ["conv_s", "resnet_s", "unet_s", "fno_s", "ptf_s"]}
rec = []
for size, R, cfgs in [("33k", R33, [(p, a, ss) for p, a, ss in CFG33]), ("small", RS, [(p, a, range(3)) for p, L in ARS.items() for a in L])]:
    for pde, a, ss in cfgs:
        for s in ss:
            b = R[(pde, a, s, "base", "best")]; o = R[(pde, a, s, "own100", "final")]; n = n1(b)
            e1 = lambda r: r["curve"][n - 1]
            d = dict(size=size, pde=pde, arch=a, s=s, excl=(size, pde, a) in EXCL, n1=n, rho=b["ratio_curve"][n - 1], lam=b["lyap_f"] / b["lyap_P"],
                     own_f=e1(o) / e1(b), c=o["onestep"] / b["onestep"], rho_after=o["ratio_curve"][n - 1], b=b)
            for arm in ["own100", "rand100"]:
                d[f"{arm}_b"] = e1(R[(pde, a, s, arm, "best")]) / e1(b)
                d[f"lam_{arm}"] = R[(pde, a, s, arm, "best")]["lyap_f"] / R[(pde, a, s, arm, "best")]["lyap_P"]
            src = RN if (size == "33k" and pde == "ks_L60" and a == "resnet") else (RS if size == "small" else None)
            if src is not None and (pde, a, s, "noise1", "best") in src:
                r = src[(pde, a, s, "noise1", "best")]
                d["noise1_b"] = e1(r) / e1(b); d["noise1_f"] = e1(src[(pde, a, s, "noise1", "final")]) / e1(b)
                d["lam_noise1"] = r["lyap_f"] / r["lyap_P"]
            d["lam_start"] = d["lam"]
            rec.append(d)
K = [r for r in rec if not r["excl"]]
facts = dict(seeds=len(K), configs=len({(r["size"], r["pde"], r["arch"]) for r in K}))

ARCC = {"conv": "#52514e", "resnet": "#2a78d6", "unet": "#1baf7a", "fno": "#eb6834", "ptf": "#8a5cc2"}
ARCN = {"conv": "CNN", "resnet": "ResNet", "unet": "U-Net", "fno": "FNO", "ptf": "Transformer"}
SZ = {"small": "8--10k", "33k": "33k"}
ORDER = {"ks_L60": 0, "gks_d0.2": 1, "cgle_phaseL1": 2, "nik_r0.02": 3}
AORD = {"conv": 0, "resnet": 1, "unet": 2, "fno": 3, "ptf": 4}


def npz_base(size, pde, a, s):
    roots = ROOTS33 if size == "33k" else [os.path.join(RES, "small")]
    for root in roots:
        for src in os.listdir(root):
            p = f"{root}/{src}/own/res/own__finetune__{pde}__{a}__s{s}__base__best.npz"
            if os.path.exists(p):
                return np.load(p)
    raise FileNotFoundError((size, pde, a, s))


def gmc(x):
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    with np.errstate(all="ignore"):
        return np.exp(np.nanmean(np.log(x), 0))


def cfgs_of(size):
    c = sorted({(r["pde"], r["arch"]) for r in K if r["size"] == size}, key=lambda t: (ORDER[t[0]], AORD[AK[t[1]]]))
    return c


def block_header(size, ncol):
    n_c = len(cfgs_of(size)); n_s = sum(r["size"] == size for r in K)
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{SZ[size]} parameters ({n_c} configurations, {n_s} seeds)}}}} \\\\"


