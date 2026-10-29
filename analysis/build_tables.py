"""Tables 4 and 5 of the appendix, written as LaTeX rows to figures/t_app_configs.tex and figures/t_app_bins.tex."""
import numpy as np
from res_common import *

NM = {"conv": "CNN", "resnet": "ResNet", "unet": "U-Net", "fno": "FNO", "ptf": "Transformer"}
def f(x):
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"

# Table 4: per configuration, ordered by median rho within each size
L = []
for size, lab in [("small", "8--10k parameters"), ("33k", "33k parameters")]:
    G = [r for r in K if r["size"] == size]
    cf = sorted({(r["pde"], r["arch"]) for r in G}, key=lambda c: np.median([r["rho"] for r in G if (r["pde"], r["arch"]) == c]))
    L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{lab} ({len(cf)} configurations, {len(G)} seeds)}}}} \\\\")
    for c in cf:
        g = [r for r in G if (r["pde"], r["arch"]) == c]
        rho = np.median([r["rho"] for r in g]); rr = gm([r["own_f"] for r in g]); imp = sum(r["own_f"] < 0.9 for r in g)
        L.append(f"{PN[c[0]]} {NM[AK[c[1]]]} & {len(g)} & {f(rho)} & {f(rr)} & {imp}/{len(g)} \\\\")
    L.append("\\midrule")
write("t_app_configs.tex", L[:-1])

# Table 5: improved / total and geometric-mean error ratio by rho bin, final checkpoints
BINS = [("le125", lambda x: x <= 1.25), ("mid", lambda x: 1.25 < x < 4), ("ge4", lambda x: x >= 4)]
R = {"small": RS, "33k": R33}
def ratio(r, arm):
    if arm == "noise1":
        return r.get("noise1_f")
    b = r["b"]; n = r["n1"]; return R[r["size"]][(r["pde"], r["arch"], r["s"], arm, "final")]["curve"][n - 1] / b["curve"][n - 1]
L = []
for size, arms in [("small", ["mse", "noise1", "rand100", "own100"]), ("33k", ["rand100", "own100"])]:
    for arm in arms:
        cells = []
        for _, sel in BINS:
            v = [ratio(r, arm) for r in K if r["size"] == size and sel(r["rho"])]
            v = [x for x in v if x is not None]
            cells.append(f"{sum(x < 0.9 for x in v)}/{len(v)} ({gm(v):.2g})")
        L.append(f"{size} {arm} & " + " & ".join(cells) + " \\\\")
write("t_app_bins.tex", L)
print(open(os.path.join(OUT, "t_app_configs.tex")).read()); print(open(os.path.join(OUT, "t_app_bins.tex")).read())
