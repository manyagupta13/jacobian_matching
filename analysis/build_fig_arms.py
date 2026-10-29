"""Figure 4: per-seed error ratio of EDJM against RJM (seeds with rho > 1.25, validation-selected checkpoints)."""
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



def paired(x, y):
    x, y = np.array(x), np.array(y); tie = np.isclose(x, y, rtol=1e-3)
    w, n = int(((x < y) & ~tie).sum()), int((~tie).sum()); return w, n, sign_p(w, n), gm(x / y)
hs = [r for r in K if r["rho"] > 1.25]

fig = plt.figure(figsize=(4.8, 1.6))
ax = fig.add_axes([0.095, 0.205, 0.30, 0.775])
lim = (4e-3, 2.5)
ax.fill_between(lim, [lim[0]] * 2, lim, color="#eaf6f0", zorder=0)
ax.plot(lim, lim, color=GREY, lw=0.7, ls="--")
lines = []
for size in ["small", "33k"]:
    g = [r for r in hs if r["size"] == size]
    for r in g:
        kw = dict(marker=MK[AK[r["arch"]]], s=14, zorder=3)
        if size == "small":
            ax.scatter(r["rand100_bst"], r["own100_bst"], color=PC[r["pde"]], edgecolor="white", lw=0.3, **kw)
        else:
            ax.scatter(r["rand100_bst"], r["own100_bst"], facecolor="none", edgecolor=PC[r["pde"]], lw=0.75, **kw)
    w, n, p, _ = paired([r["own100_bst"] for r in g], [r["rand100_bst"] for r in g])
    lines.append(f"{SZ[size]}: EDJM better on {w}/{n} seeds")
ax.text(0.96, 0.04, "EDJM better", transform=ax.transAxes, fontsize=6.4, color="#1a7a55", ha="right", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("error ratio $r$, RJM", labelpad=1); ax.set_ylabel("error ratio $r$, EDJM", labelpad=1); clean(ax, False)
ax.tick_params(pad=1.5)
h = [Line2D([], [], marker="o", ls="", color=PC[p], ms=4.5, label=PN[p]) for p in ["ks_L60", "gks_d0.2", "cgle_phaseL1", "nik_r0.02"]]
hs_ = [Line2D([], [], marker="o", ls="", color="#6b6a66", ms=4.5, label="8–10k"), Line2D([], [], marker="o", ls="", markerfacecolor="none", color="#6b6a66", ms=4.5, label="33k")]
h2 = [Line2D([], [], marker=MK[k], ls="", color="#6b6a66", ms=4.5, label=ARCN[k]) for k in ["conv", "resnet", "unet", "fno", "ptf"]]
kw = dict(frameon=False, fontsize=6.4, handletextpad=0.1, labelspacing=0.35, borderaxespad=0, title_fontsize=6.6, alignment="left")
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.44, 0.98), title="system", **kw)
fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.62, 0.98), title="architecture", **kw)
fig.legend(handles=hs_, loc="upper left", bbox_to_anchor=(0.86, 0.98), title="size", **kw)
fig.text(0.44, 0.04, "\n".join(lines), fontsize=6.6, color=INK, ha="left", va="bottom")
fig.savefig(f"{OUT}/fig_arms.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/fig_arms.png", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
