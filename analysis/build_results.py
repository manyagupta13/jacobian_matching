"""Figures 1-3, Table 5/6 numbers and every number quoted in Section 5. Writes figures/ and figures/results_facts.json."""
import os, json, numpy as np
from res_common import *   # records K (74 seeds), R33, RS, RN, helpers, style, PN, AN, AK, MK, PC, TW
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SZ = {"small": "8–10k", "33k": "33k"}
ORDER = {"ks_L60": 0, "gks_d0.2": 1, "cgle_phaseL1": 2, "nik_r0.02": 3}
AORD = {"conv": 0, "resnet": 1, "unet": 2, "fno": 3, "ptf": 4}
ARCN = {"conv": "CNN", "resnet": "ResNet", "unet": "U-Net", "fno": "FNO", "ptf": "Transformer"}
GREY, BLUE, ORANGE, GREEN, PURPLE, INK = "#9a9994", "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#2b2b29"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

# extra per-seed fields: control arm and best/final for every arm
for r in K:
    R = R33 if r["size"] == "33k" else RS
    b = R[(r["pde"], r["arch"], r["s"], "base", "best")]; n = r["n1"]; e1 = lambda x: x["curve"][n - 1] / b["curve"][n - 1]
    for arm in ["mse", "rand100", "own100"]:
        r[f"{arm}_fin"] = e1(R[(r["pde"], r["arch"], r["s"], arm, "final")])
        r[f"{arm}_bst"] = e1(R[(r["pde"], r["arch"], r["s"], arm, "best")])
    if "noise1_b" in r:
        r["noise1_bst"], r["noise1_fin"] = r["noise1_b"], r["noise1_f"]
facts = {}
NICE = {'rand100': 'RJM', 'noise1': 'Noise', 'mse': 'Cont. training', 'own100': 'EDJM'}
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def npz_base(size, pde, a, s):
    roots = ROOTS33 if size == "33k" else [os.path.join(RES, "small")]
    for root in roots:
        for src in os.listdir(root):
            p = f"{root}/{src}/own/res/own__finetune__{pde}__{a}__s{s}__base__best.npz"
            if os.path.exists(p):
                return np.load(p)


def gmc(x):
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    with np.errstate(all="ignore"):
        return np.exp(np.nanmean(np.log(x), 0))


def clean(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y" if grid_y else "both", which="major", lw=0.4, color="#e6e5e1"); ax.set_axisbelow(True)


# ================================================================ Fig 1: the ledger, one system, two architectures
pairs = [("gks_d0.2", "resnet_s", "gKS ResNet"), ("gks_d0.2", "conv_s", "gKS CNN")]
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.25), sharey=True)
for ax, (pde, a, title) in zip(axs, pairs):
    E, P = [], []
    for s in range(3):
        z = npz_base("small", pde, a, s); lam = float(z["lam"])
        E.append(gmc(z["rel_true"])); P.append(gmc(z["rel_P"]))
    t = np.arange(1, len(E[0]) + 1) * lam; m = t <= 1.3
    E, P = np.array(E)[:, m], np.array(P)[:, m]; t = t[m]
    Em, Pm = np.exp(np.log(E).mean(0)), np.exp(np.log(P).mean(0))
    ax.fill_between(t, Pm, Em, where=Em > Pm, color=ORANGE, alpha=0.13, lw=0)
    for e, p in zip(E, P):
        ax.plot(t, e, color=INK, lw=0.5, alpha=0.35); ax.plot(t, p, color=BLUE, lw=0.5, alpha=0.35, ls="--")
    ax.plot(t, Em, color=INK, lw=1.6, label=r"actual error $\|e_n\|$")
    ax.plot(t, Pm, color=BLUE, lw=1.6, ls="--", label=r"counterfactual error $\|e^P_n\|$")
    ax.axvline(1, color=GREY, lw=0.7, ls=":")
    ax.set_yscale("log"); ax.set_xlim(0, 1.3); ax.set_ylim(3e-3, 2); ax.set_title(title)
    ax.set_xlabel("time [Lyapunov times]"); clean(ax)
    k = np.argmin(abs(t - 1)); facts[f"fig1_{a}"] = dict(e=float(Em[k]), eP=float(Pm[k]), rho=float(Em[k] / Pm[k]))
    ax.set_title(title + rf", $\rho={Em[k] / Pm[k]:.1f}$ at 1 LT")
axs[1].annotate("excess growth\nfrom $J_f\\neq J_P$", xy=(0.75, 0.2), xytext=(0.12, 0.6), fontsize=7, color="#9c4a1f",
                arrowprops=dict(arrowstyle="-", color="#9c4a1f", lw=0.6))
axs[0].set_ylabel("relative error"); axs[0].legend(frameon=False, loc="lower right")
fig.tight_layout(w_pad=1.5); fig.savefig(f"{OUT}/fig_ledger.pdf"); plt.close(fig)

# ================================================================ Fig 2: the diagnostic, 8-10k | 33k
BINS = [("low", lambda r: r <= 1.25), ("mid", lambda r: 1.25 < r < 4), ("high", lambda r: r >= 4)]
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.9), sharey=True)
for ax, size in zip(axs, ["small", "33k"]):
    g = [r for r in K if r["size"] == size]
    ax.axvspan(0.2, 1.25, color="#efeee9", zorder=0); ax.axvspan(4, 100, color="#e8f1fb", zorder=0)
    for r in g:
        ax.scatter(min(r["rho"], 70), r["own_f"], color=PC[r["pde"]], marker=MK[AK[r["arch"]]], s=17,
                   edgecolor="white", lw=0.35, zorder=3)
    ax.axhline(1, color=INK, lw=0.7); ax.axhline(0.9, color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.3, 90); ax.set_ylim(4e-3, 8)
    for lab, sel in BINS:
        gg = [r for r in g if sel(r["rho"])]; k = sum(r["own_f"] < 0.9 for r in gg)
        facts[f"bin_{size}_{lab}"] = [k, len(gg), gm([r["own_f"] for r in gg]) if gg else None]
        if lab != "mid":
            x = 0.62 if lab == "low" else 19
            ax.text(x, 2.4, f"{k}/{len(gg)}\nimproved", ha="center", va="bottom", fontsize=6.8, color=INK, linespacing=1.0)
    ax.set_title(f"{SZ[size]} parameters\n{len({(r['pde'], r['arch']) for r in g})} configurations, {len(g)} seeds", fontsize=8)
    ax.set_xlabel(r"amplification ratio $\rho$ of the starting model"); clean(ax, False)
axs[0].set_ylabel("error ratio $r$ after EDJM")
h = [Line2D([], [], marker="o", ls="", color=PC[p], ms=4.5, label=PN[p]) for p in ["ks_L60", "gks_d0.2", "cgle_phaseL1", "nik_r0.02"]]
h += [Line2D([], [], marker=MK[k], ls="", color="#6b6a66", ms=4.5, label=ARCN[k]) for k in ["conv", "resnet", "unet", "fno", "ptf"]]
fig.legend(handles=h, frameon=False, ncol=9, loc="upper center", handletextpad=0.1, columnspacing=0.9, fontsize=6.8)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2); fig.savefig(f"{OUT}/fig_diagnostic.pdf"); plt.close(fig)

# control and other arms by bin (final checkpoint)
for size in ["small", "33k"]:
    for lab, sel in BINS:
        gg = [r for r in K if r["size"] == size and sel(r["rho"])]
        for arm in ["mse", "rand100", "own100"]:
            v = [r[f"{arm}_fin"] for r in gg]
            facts[f"arm_{size}_{lab}_{arm}"] = [int(sum(x < 0.9 for x in v)), len(v), gm(v) if v else None]
        v = [r["noise1_fin"] for r in gg if "noise1_fin" in r]
        if v:
            facts[f"arm_{size}_{lab}_noise1"] = [int(sum(x < 0.9 for x in v)), len(v), gm(v)]

# ================================================================ Fig 3: the ceiling
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.95))
def pt(ax, r, x, y):
    kw = dict(marker=MK[AK[r["arch"]]], s=16, zorder=3)
    if r["size"] == "small":
        ax.scatter(x, y, color=PC[r["pde"]], edgecolor="white", lw=0.35, **kw)
    else:
        ax.scatter(x, y, facecolor="none", edgecolor=PC[r["pde"]], lw=0.8, **kw)
for r in K:
    pt(axs[0], r, min(r["rho"], 70), r["rho_after"]); pt(axs[1], r, r["c"] / r["rho"], r["own_f"])
ax = axs[0]
ax.plot([0.25, 90], [0.25, 90], color=GREY, lw=0.7, ls="--"); ax.axhline(1, color=INK, lw=0.7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.25, 90); ax.set_ylim(0.25, 90); clean(ax, False)
ax.set_xlabel(r"$\rho$ before fine-tuning"); ax.set_ylabel(r"$\rho$ after EDJM"); ax.set_title("(a) fine-tuning removes the excess growth")
ax.text(12, 25, "unchanged", rotation=40, fontsize=6.8, color=GREY, ha="center")
ax = axs[1]
lim = (4e-3, 6); ax.plot(lim, lim, color=GREY, lw=0.7, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.axhline(1, color=INK, lw=0.7); ax.set_xscale("log"); ax.set_yscale("log"); clean(ax, False)
ax.set_xlabel(r"one-step cost / amplification ratio, $c/\rho$"); ax.set_ylabel("error ratio $r$ after EDJM"); ax.set_title(r"(b) the gain follows $c/\rho$")
ax.text(0.9, 0.06, "$r = c/\\rho$", rotation=40, fontsize=6.8, color=GREY)
h = [Line2D([], [], marker="o", ls="", color=PC[p], ms=4.5, label=PN[p]) for p in ["ks_L60", "gks_d0.2", "cgle_phaseL1", "nik_r0.02"]]
h += [Line2D([], [], marker="o", ls="", color="#6b6a66", ms=4.5, label="8–10k"), Line2D([], [], marker="o", ls="", markerfacecolor="none", color="#6b6a66", ms=4.5, label="33k")]
h2 = [Line2D([], [], marker=MK[k], ls="", color="#6b6a66", ms=4.5, label=ARCN[k]) for k in ["conv", "resnet", "unet", "fno", "ptf"]]
fig.legend(handles=h, frameon=False, ncol=6, loc="upper center", fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
fig.legend(handles=h2, frameon=False, ncol=5, loc="upper center", fontsize=6.8, bbox_to_anchor=(0.5, 0.945))
fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.0); fig.savefig(f"{OUT}/fig_ceiling.pdf"); plt.close(fig)
from scipy.stats import spearmanr
P = np.log([r["c"] / r["rho"] for r in K]); Y = np.log([r["own_f"] for r in K])
for sz in ["small", "33k"]:
    m = np.array([r["size"] == sz for r in K])
    facts[f"corr_{sz}"] = [float(np.corrcoef(P[m], Y[m])[0, 1]), float(spearmanr(P[m], Y[m])[0]), int(m.sum())]
    facts[f"rho_after_{sz}"] = float(np.median([r["rho_after"] for r in K if r["size"] == sz]))
    facts[f"c_help_{sz}"] = float(np.median([r["c"] for r in K if r["size"] == sz and r["rho"] > 1.25]))
    facts[f"c_low_{sz}"] = float(np.median([r["c"] for r in K if r["size"] == sz and r["rho"] <= 1.25]))

# ================================================================ Paired comparisons (Table 6); Figure 4 is drawn by build_fig_arms.py
hs = [r for r in K if r["rho"] > 1.25]

def paired(x, y):
    x, y = np.array(x), np.array(y); tie = np.isclose(x, y, rtol=1e-3)
    w, n = int(((x < y) & ~tie).sum()), int((~tie).sum()); return w, n, sign_p(w, n), gm(x / y)


for size in ["small", "33k"]:
    g = [r for r in hs if r["size"] == size]
    for ck in ["bst", "fin"]:
        facts[f"own_vs_rand_{size}_{ck}"] = paired([r[f"own100_{ck}"] for r in g], [r[f"rand100_{ck}"] for r in g])
        facts[f"own_vs_mse_{size}_{ck}"] = paired([r[f"own100_{ck}"] for r in g], [r[f"mse_{ck}"] for r in g])
        gn = [r for r in g if f"noise1_{ck}" in r]
        facts[f"own_vs_noise_{size}_{ck}"] = paired([r[f"own100_{ck}"] for r in gn], [r[f"noise1_{ck}"] for r in gn])
    facts[f"n_help_{size}"] = len(g)
json.dump(facts, open(f"{OUT}/results_facts.json", "w"), indent=1, default=float)
for k, v in facts.items():
    print(k, v)

