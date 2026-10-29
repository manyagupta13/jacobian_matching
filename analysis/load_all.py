"""Load every own__*.npz result from all extracted archives into one table of per-run metrics."""
import glob, os, json, numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ROOTS = [os.path.join(RESULTS, "33k_round1"), os.path.join(RESULTS, "33k")]   # order matters: duplicates keep the first


def gmean(x, axis=0):
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    with np.errstate(all="ignore"):
        return np.exp(np.nanmean(np.log(x), axis))


def metrics(z):
    S = z["rel_true"].shape[1]
    rt = gmean(z["rel_true"])
    with np.errstate(all="ignore"):
        ratio = gmean(z["rel_true"] / z["rel_P"])
    lam = float(z["lam"])
    r = dict(err50=rt[49], err100=rt[99], err200=rt[S - 1], onestep=float(gmean(z["d_rel"].reshape(-1))),
             own_gap=float(z["own_gap_rel"]), rand_gap=float(z["rand_gap_rel"]), own_1mcos=float(z["own_1mcos"]),
             vpt=float(np.median(z["vpt_steps"])) * lam, vpt_cap=S * lam, ndiv=int((z["div_step"] <= S).sum()),
             ntraj=int(z["rel_true"].shape[0]), tp25=ratio[24], tp50=ratio[49], tp100=ratio[99],
             lyap_f=float(np.nanmedian(z["lyap_f"][:, 0])) if "lyap_f" in z else np.nan,
             lyap_P=float(np.nanmedian(z["lyap_P"][:, 0])) if "lyap_P" in z else np.nan,
             curve=rt, ratio_curve=ratio)
    return r


def load():
    recs = {}
    for root in ROOTS:
        for p in sorted(glob.glob(os.path.join(root, "*", "own", "res", "own__*.npz"))):
            b = os.path.basename(p)[:-4].split("__")
            key = (b[2], b[3], int(b[4][1:]), b[5], b[6])     # pde, arch, seed, arm, ck
            if key in recs:                                    # shared runs appear in several archives: keep the first
                continue
            with np.load(p) as z:
                recs[key] = metrics({k: z[k] for k in z.files})
                recs[key]["src"] = p.split("/")[-4]
    hists = {}
    for root in ROOTS:
        for p in glob.glob(os.path.join(root, "*", "own", "hist", "own__*.json")):
            b = os.path.basename(p)[:-5].split("__")
            hists.setdefault((b[2], b[3], int(b[4][1:]), b[5]), json.load(open(p)))
    return recs, hists
