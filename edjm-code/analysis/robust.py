import os, sys, json, numpy as np
from res_common import *
out = {}
for size in ["small", "33k"]:
    G = [r for r in K if r["size"] == size]
    for frac in [0.5, 1.0, 1.5]:
        rows = []
        for r in G:
            R = R33 if size == "33k" else RS
            b = r["b"]; o = R[(r["pde"], r["arch"], r["s"], "own100", "final")]
            n = int(round(frac * r["n1"]))
            n = min(n, len(b["curve"]))
            rho = b["ratio_curve"][n - 1]; rr = o["curve"][n - 1] / b["curve"][n - 1]
            rows.append((rho, rr))
        rows = np.array(rows)
        for lo in [1.1, 1.25, 1.5]:
            for hi in [3, 4, 5]:
                L = rows[rows[:, 0] <= lo]; H = rows[rows[:, 0] >= hi]
                out[f"{size}|{frac}|{lo}|{hi}"] = [int((L[:, 1] < 0.9).sum()), len(L), int((H[:, 1] < 0.9).sum()), len(H)]
for k, v in out.items(): print(k, v)
json.dump(out, open(os.path.join(OUT, "robust.json"), "w"), indent=1)
