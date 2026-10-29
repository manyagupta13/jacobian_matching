"""Appendix: ledger vs actual error for every configuration. Usage: OUT=<dir> python build_appendix_ledger.py"""
import os, numpy as np
from res_common import *
from matplotlib.lines import Line2D
INK, BLUE, ORANGE, GREY = "#2b2b29", "#2a78d6", "#eb6834", "#9a9994"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
ARCN = {"conv": "CNN", "resnet": "ResNet", "unet": "U-Net", "fno": "FNO", "ptf": "Transformer"}


def npz_base(size, pde, a, s):
    roots = ROOTS33 if size == "33k" else [os.path.join(RES, "small")]
    for root in roots:
        for src in os.listdir(root):
            p = f"{root}/{src}/own/res/own__finetune__{pde}__{a}__s{s}__base__best.npz"
            if os.path.exists(p):
                return np.load(p)
    return None


def gmc(x):
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    with np.errstate(all="ignore"):
        return np.exp(np.nanmean(np.log(x), 0))


def panel(ax, size, pde, a, seeds, excluded=False):
    E, P, lam = [], [], None
    for s in seeds:
        z = npz_base(size, pde, a, s)
        if z is None:
            continue
        lam = float(z["lam"]); E.append(gmc(z["rel_true"])); P.append(gmc(z["rel_P"]))
    t = np.arange(1, len(E[0]) + 1) * lam; m = t <= 1.3; t = t[m]
    E, P = np.array(E)[:, m], np.array(P)[:, m]
    with np.errstate(all="ignore"):
        Em, Pm = np.exp(np.nanmean(np.log(E), 0)), np.exp(np.nanmean(np.log(P), 0))
    a_ = 0.35 if not excluded else 0.2
    for e, p in zip(E, P):
        ax.plot(t, e, color=INK, lw=0.45, alpha=a_); ax.plot(t, p, color=BLUE, lw=0.45, alpha=a_, ls="--")
    ax.fill_between(t, Pm, Em, where=Em > Pm, color=ORANGE, alpha=0.12 if not excluded else 0.05, lw=0)
    ax.plot(t, Em, color=INK if not excluded else GREY, lw=1.2); ax.plot(t, Pm, color=BLUE if not excluded else GREY, lw=1.2, ls="--")
    ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_xlim(0, 1.3); ax.set_ylim(1e-3, 20)
    rho = [r["rho"] for r in K if (r["size"], r["pde"], r["arch"]) == (size, pde, a)]
    lab = "excluded (diverges)" if excluded else rf"$\rho={np.median(rho):.2g}$"
    ax.text(0.04, 0.93, lab, transform=ax.transAxes, va="top", fontsize=6.8, color=INK if not excluded else GREY,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(True, axis="y", lw=0.35, color="#e6e5e1"); ax.set_axisbelow(True)


def legend(fig):
    h = [Line2D([], [], color=INK, lw=1.2, label=r"actual error $\|e_n\|$"), Line2D([], [], color=BLUE, lw=1.2, ls="--", label=r"counterfactual error $\|e^P_n\|$"),
         Line2D([], [], color=INK, lw=0.45, alpha=0.5, label="individual seeds")]
    fig.legend(handles=h, frameon=False, ncol=3, loc="upper center", fontsize=7)


# ---------------------------------------------------------------- 8-10k: rows = PDEs, columns = architectures
PDES = ["ks_L60", "gks_d0.2", "cgle_phaseL1"]
COLS = ["conv", "resnet", "unet", "fno", "ptf"]
ARCH_OF = {("gks_d0.2", "fno"): "fno32_s"}
fig, axs = plt.subplots(3, 5, figsize=(TW, 4.6), sharex=True, sharey=True)
for i, pde in enumerate(PDES):
    for j, k in enumerate(COLS):
        a = ARCH_OF.get((pde, k), f"{k}_s" if k != "conv" else "conv_s")
        if k == "ptf": a = "ptf_s"
        if k == "unet": a = "unet_s"
        if k == "resnet": a = "resnet_s"
        if k == "fno" and (pde, k) not in ARCH_OF: a = "fno_s"
        panel(axs[i, j], "small", pde, a, range(3), excluded=(pde == "ks_L60" and k == "conv"))
        if i == 0:
            axs[i, j].set_title(ARCN[k])
    axs[i, 0].set_ylabel(f"{PN[pde]}\nrelative error")
for j in range(5):
    axs[-1, j].set_xlabel("time [LT]")
legend(fig); fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.4)
fig.savefig(f"{OUT}/fig_app_ledger_small.pdf"); plt.close(fig)

# ---------------------------------------------------------------- 33k: rows = architectures, columns = PDEs
PDES33 = ["ks_L60", "gks_d0.2", "cgle_phaseL1", "nik_r0.02"]
fig, axs = plt.subplots(2, 4, figsize=(TW, 3.3), sharex=True, sharey=True)
for i, a in enumerate(["resnet", "fno"]):
    for j, pde in enumerate(PDES33):
        ax = axs[i, j]
        if pde == "nik_r0.02" and a == "fno":
            ax.axis("off"); ax.text(0.5, 0.5, "no fine-tuning runs", ha="center", va="center", color=GREY, transform=ax.transAxes, fontsize=7)
            continue
        seeds = [r["s"] for r in K if (r["size"], r["pde"], r["arch"]) == ("33k", pde, a)]
        panel(ax, "33k", pde, a, seeds)
        if i == 0:
            ax.set_title(PN[pde])
    axs[i, 0].set_ylabel(f"{ARCN[a]}\nrelative error")
for j in range(3):
    axs[-1, j].set_xlabel("time [LT]")
axs[0, 3].set_xlabel("time [LT]"); axs[0, 3].tick_params(labelbottom=True)
legend(fig); fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.4)
fig.savefig(f"{OUT}/fig_app_ledger_33k.pdf"); plt.close(fig)
print("ok")
