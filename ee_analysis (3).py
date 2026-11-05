"""
ee_analysis.py - router comparison, oracle savings, Pareto curves, cost-transfer study, multi-seed aggregation.

Protocol (this is what makes the comparison defensible):
  * thresholds are chosen on VALIDATION only, for each router and each accuracy budget
  * the chosen thresholds are then evaluated on TEST
  * accuracy drop is measured against the baseline (teacher) test accuracy
  * latency = expected relative latency from measured stage costs (full network = 1.0); use ee_onnx.py for real ms
"""
from __future__ import annotations
import csv, json, os
import numpy as np

from ee_core import ALL_ROUTERS
from ee_train import (collect, model_accuracy, simulate, make_grids, operating_points, best_under_budget)

DEFAULT_BUDGETS = (0.25, 0.5, 1.0, 2.0)


# ----------------------------------------------------------------------------- oracle
def oracle_stats(P, Y, costs):
    """Upper bounds that no router can beat, given the trained exit heads.
    iso   : exit at the earliest head that is correct, but ONLY if the final head is also correct (accuracy == baseline head)
    free  : exit at the earliest correct head (accuracy can exceed the baseline: it also rescues samples the final head misses)"""
    N, K1 = P.shape
    K = K1 - 1
    correct = P == Y[:, None]
    costs = np.asarray(costs)
    first = np.where(correct[:, :K].any(1), correct[:, :K].argmax(1), K)
    stage_iso = np.where(correct[:, K], first, K)
    stage_free = first
    out = {}
    for name, st in (("oracle_iso", stage_iso), ("oracle_free", stage_free)):
        fr = np.bincount(st, minlength=K1) / N
        acc = float(correct[np.arange(N), st].mean() * 100)
        out[name] = dict(acc=acc, fr=fr.tolist(), lat=float((fr * costs).sum()))
    return out


# ----------------------------------------------------------------------------- selection on val, evaluation on test
def select_and_eval(val, test, costs, ref_val, ref_test, budgets=DEFAULT_BUDGETS, routers=None,
                    grid_mode="quantile", grid_n=60, costs_select=None):
    """For each router and budget: pick thresholds on val (min latency s.t. val_acc >= ref_val - budget), score on test.
    costs_select: costs used when PICKING thresholds (default = costs). Latency is always reported with `costs`."""
    costs_select = costs if costs_select is None else costs_select
    orc = oracle_stats(test["P"], test["Y"], costs)
    orc_saving = 1.0 - orc["oracle_iso"]["lat"]
    rows = []
    for r in (routers or [r for r in ALL_ROUTERS if r in val["S"]]):
        Sv, St = val["S"][r], test["S"][r]
        pts = operating_points(val["P"], Sv, val["Y"], costs_select, make_grids(Sv, grid_mode, grid_n))
        for b in budgets:
            (thr, vacc, _, vlat), feasible = best_under_budget(pts, ref_val, b)
            tacc, tfr, tlat = simulate(test["P"], St, test["Y"], thr, costs)
            saving = 1.0 - tlat
            rows.append(dict(router=r, budget=float(b), feasible=bool(feasible), thr=[float(t) for t in thr],
                             val_acc=vacc, val_rel_lat=vlat, test_acc=tacc, test_drop=ref_test - tacc, rel_lat=tlat,
                             saving_pct=100 * saving, exit_fracs=[float(f) for f in tfr],
                             oracle_capture_pct=(100 * saving / orc_saving) if orc_saving > 1e-9 else float("nan")))
    for name in ("oracle_iso", "oracle_free"):
        o = orc[name]
        rows.append(dict(router=name, budget=float("nan"), feasible=True, thr=[], val_acc=float("nan"), val_rel_lat=float("nan"), test_acc=o["acc"],
                         test_drop=ref_test - o["acc"], rel_lat=o["lat"], saving_pct=100 * (1 - o["lat"]),
                         exit_fracs=o["fr"], oracle_capture_pct=100.0 if name == "oracle_iso" else float("nan")))
    return rows


def format_rows(rows):
    h = f"{'router':<12}{'budget':>7}{'test acc':>9}{'drop':>7}{'relLat':>8}{'saving%':>9}{'oracle%':>9}  exits%"
    lines = [h, "-" * len(h)]
    for r in rows:
        ex = " ".join(f"{100 * f:5.1f}" for f in r["exit_fracs"])
        flag = "" if r["feasible"] else "  (budget infeasible -> never-exit)"
        lines.append(f"{r['router']:<12}{r['budget']:>7.2f}{r['test_acc']:>9.2f}{r['test_drop']:>7.2f}{r['rel_lat']:>8.3f}"
                     f"{r['saving_pct']:>9.1f}{r['oracle_capture_pct']:>9.1f}  {ex}{flag}")
    return "\n".join(lines)


def save_rows(rows, path_prefix):
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    with open(path_prefix + ".json", "w") as f: json.dump(rows, f, indent=1)
    keys = ["router", "budget", "feasible", "test_acc", "test_drop", "val_acc", "val_rel_lat", "rel_lat", "saving_pct", "oracle_capture_pct", "thr", "exit_fracs"]
    with open(path_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows: w.writerow([r[k] for k in keys])


# ----------------------------------------------------------------------------- multi-seed aggregation
def aggregate_seeds(rows_per_seed):
    """rows_per_seed: list (one entry per seed) of select_and_eval() outputs -> mean/std per (router, budget)."""
    keys = [(r["router"], r["budget"]) for r in rows_per_seed[0]]
    out = []
    for k in keys:
        sel = [next(r for r in rows if (r["router"], r["budget"]) == k or (r["router"] == k[0] and np.isnan(r["budget"]) and np.isnan(k[1])))
               for rows in rows_per_seed]
        d = dict(router=k[0], budget=k[1], n_seeds=len(sel))
        for m in ("test_acc", "test_drop", "rel_lat", "saving_pct", "oracle_capture_pct"):
            v = np.array([s[m] for s in sel], dtype=float)
            d[m + "_mean"], d[m + "_std"] = float(np.nanmean(v)), float(np.nanstd(v))
        out.append(d)
    return out


def format_aggregate(agg):
    h = f"{'router':<12}{'budget':>7}{'test acc':>16}{'relLat':>16}{'saving%':>16}"
    lines = [h, "-" * len(h)]
    for a in agg:
        lines.append(f"{a['router']:<12}{a['budget']:>7.2f}{a['test_acc_mean']:>9.2f}±{a['test_acc_std']:<5.2f}"
                     f"{a['rel_lat_mean']:>10.3f}±{a['rel_lat_std']:<5.3f}{a['saving_pct_mean']:>10.1f}±{a['saving_pct_std']:<4.1f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- hardware cost transfer
def cost_transfer(val, test, costs_a, costs_b, ref_val, ref_test, budget=1.0, routers=None, **kw):
    """Pick thresholds with hardware-A costs, deploy on hardware B. Compare with thresholds picked using B's own costs.
    'regret' = extra relative latency paid on B by using A's costs (only matters with >= 2 exits)."""
    ra = select_and_eval(val, test, costs_b, ref_val, ref_test, (budget,), routers, costs_select=costs_a, **kw)
    rb = select_and_eval(val, test, costs_b, ref_val, ref_test, (budget,), routers, **kw)
    out = []
    for a, b in zip(ra, rb):
        if a["router"].startswith("oracle"): continue
        out.append(dict(router=a["router"], lat_with_A_thr=a["rel_lat"], lat_with_B_thr=b["rel_lat"],
                        regret=a["rel_lat"] - b["rel_lat"], acc_A=a["test_acc"], acc_B=b["test_acc"]))
    return out


# ----------------------------------------------------------------------------- plots
def plot_pareto(curves, ref_test, oracle, path, title=""):
    """curves: {router: rows from select_and_eval with many budgets}. x = rel. latency, y = test acc."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r, rows in curves.items():
        rows = sorted(rows, key=lambda d: d["rel_lat"])
        ax.plot([d["rel_lat"] for d in rows], [d["test_acc"] for d in rows], marker="o", ms=3, lw=1.2, label=r)
    ax.scatter([1.0], [ref_test], c="k", marker="s", zorder=5, label="baseline")
    ax.scatter([oracle["oracle_iso"]["lat"]], [oracle["oracle_iso"]["acc"]], c="gold", edgecolors="k", marker="*", s=180, zorder=6, label="oracle (iso-acc)")
    ax.set_xlabel("expected relative latency (full network = 1.0)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- one-call report
def full_report(res, val_loader, test_loader, out_dir=None, budgets=DEFAULT_BUDGETS, hw=None, verbose=True):
    """Collect val/test, run the router comparison + oracle + Pareto plot, write csv/json/png. Returns a dict."""
    out_dir = out_dir or res.cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    dev, model = res.device, res.model.eval()
    hw = hw or res.cfg["cost_target"]
    costs = res.costs[hw] if hw in res.costs else res.train_costs
    val, test = collect(model, val_loader, dev), collect(model, test_loader, dev)
    ref_val, ref_test = model_accuracy(res.teacher, val_loader, dev), model_accuracy(res.teacher, test_loader, dev)

    rows = select_and_eval(val, test, costs, ref_val, ref_test, budgets)
    dense = np.round(np.linspace(0.0, 5.0, 26), 3)
    curves = {}
    for row in select_and_eval(val, test, costs, ref_val, ref_test, dense):
        if not row["router"].startswith("oracle"): curves.setdefault(row["router"], []).append(row)
    orc = oracle_stats(test["P"], test["Y"], costs)
    plot_pareto(curves, ref_test, orc, f"{out_dir}/{res.tag}_pareto_{hw}.png", f"{res.tag} ({hw} costs)")
    save_rows(rows, f"{out_dir}/{res.tag}_routers_{hw}")
    transfer = None
    if "cpu" in res.costs and "gpu" in res.costs:
        transfer = cost_transfer(val, test, res.costs["gpu"], res.costs["cpu"], ref_val, ref_test, budget=max(budgets))
        with open(f"{out_dir}/{res.tag}_cost_transfer_gpu_to_cpu.json", "w") as f: json.dump(transfer, f, indent=1)
    head_acc = [float((test["P"][:, k] == test["Y"]).mean() * 100) for k in range(test["P"].shape[1])]
    if verbose:
        print(f"\n=== {res.tag} | costs: {hw} {[round(c, 3) for c in costs]} | baseline test acc {ref_test:.2f}% (val {ref_val:.2f}%) | head test acc {[round(a, 2) for a in head_acc]}")
        print(format_rows(rows))
        if transfer:
            print("\nGPU-picked thresholds deployed on CPU (regret = extra rel. latency vs CPU-picked):")
            for t in transfer: print(f"  {t['router']:<12} lat {t['lat_with_A_thr']:.3f} vs {t['lat_with_B_thr']:.3f}  regret {t['regret']:+.3f}")
        print(f"\nfiles in {out_dir}: *_routers_{hw}.csv/json, *_pareto_{hw}.png")
    return dict(rows=rows, curves=curves, oracle=orc, val=val, test=test, ref_val=ref_val, ref_test=ref_test,
                costs=costs, head_test_acc=head_acc, transfer=transfer)


# ----------------------------------------------------------------------------- matched-accuracy comparison across seeds
def _frontier(rows):
    pts = sorted(((r["rel_lat"], r["test_acc"]) for r in rows))
    lat, acc, best = [], [], -1e9
    for l, a in pts:
        if a > best + 1e-9:
            lat.append(l); acc.append(a); best = a
    return np.array(lat), np.array(acc)


def latency_at_accuracy(curve_rows, target_acc):
    """Lowest relative latency a router needs to reach `target_acc` (interpolated along its validation-selected Pareto
    frontier). nan if the router never reaches that accuracy."""
    lat, acc = _frontier(curve_rows)
    if len(acc) == 0 or target_acc > acc.max() + 1e-9 or target_acc < acc.min() - 1e-9:
        return float("nan")
    return float(np.interp(target_acc, acc, lat))


def matched_accuracy_table(reports, drops=(0.5, 1.0), ref="margin"):
    """The fair router comparison. For each seed: latency each router needs to stay within `drop` points of the baseline's
    test accuracy (so every router is compared at the SAME accuracy). Reports mean +- std over seeds, plus the paired
    difference to `ref` and in how many seeds the router beats it. reports = list of full_report() outputs (one per seed)."""
    routers = [r for r in ALL_ROUTERS if r in reports[0]["curves"]]
    out = []
    for d in drops:
        sav = {r: [] for r in routers}
        for rep in reports:
            for r in routers:
                sav[r].append(100.0 * (1.0 - latency_at_accuracy(rep["curves"][r], rep["ref_test"] - d)))
        for r in routers:
            v, base = np.array(sav[r]), np.array(sav[ref])
            diff = v - base
            out.append(dict(drop=d, router=r, saving_mean=float(np.nanmean(v)), saving_std=float(np.nanstd(v)),
                diff_vs_ref_mean=float(np.nanmean(diff)), diff_vs_ref_std=float(np.nanstd(diff)),
                wins=int(np.nansum(diff > 0)), n=len(reports), ref=ref,
                valid=int(np.sum(~np.isnan(v)))))
    return out


def format_matched(rows):
    lines = []
    for d in sorted({r["drop"] for r in rows}):
        lines.append(f"\n=== latency saving % at test accuracy within {d} pt of baseline (matched accuracy), mean ± std over seeds ===")
        lines.append(f"{'router':<12}{'saving %':>16}{'vs ' + rows[0]['ref']:>20}{'wins':>8}")
        for r in (x for x in rows if x["drop"] == d):
            tag = "(reference)" if r["router"] == r["ref"] else f"{r['diff_vs_ref_mean']:+.1f} ± {r['diff_vs_ref_std']:.1f}"
            wins = "-" if r["router"] == r["ref"] else f"{r['wins']}/{r['valid']}"
            flag = f"  <-- only {r['valid']} of {r['n']} seeds reached this accuracy" if r["valid"] < r["n"] else ""
            lines.append(f"{r['router']:<12}{r['saving_mean']:>9.1f} ± {r['saving_std']:<5.1f}{tag:>20}{wins:>8}{flag}")
    return "\n".join(lines)


def plot_seed_pareto(reports, path, title=""):
    """All seeds, all routers: thin lines = individual seeds, thick = mean across seeds (same budgets)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    routers = [r for r in ALL_ROUTERS if r in reports[0]["curves"]]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, r in enumerate(routers):
        col = f"C{i}"
        for rep in reports:
            rows = sorted(rep["curves"][r], key=lambda d: d["budget"])
            ax.plot([d["rel_lat"] for d in rows], [d["test_acc"] for d in rows], color=col, alpha=0.18, lw=0.8)
        lat = np.mean([[d["rel_lat"] for d in sorted(rep["curves"][r], key=lambda d: d["budget"])] for rep in reports], axis=0)
        acc = np.mean([[d["test_acc"] for d in sorted(rep["curves"][r], key=lambda d: d["budget"])] for rep in reports], axis=0)
        o = np.argsort(lat)
        ax.plot(lat[o], acc[o], color=col, lw=2, label=r)
    ref = np.mean([rep["ref_test"] for rep in reports])
    orc = np.mean([[rep["oracle"]["oracle_iso"]["lat"], rep["oracle"]["oracle_iso"]["acc"]] for rep in reports], axis=0)
    ax.scatter([1.0], [ref], c="k", marker="s", zorder=5, label="baseline")
    ax.scatter([orc[0]], [orc[1]], c="gold", edgecolors="k", marker="*", s=200, zorder=6, label="oracle (iso-acc)")
    ax.set_xlabel("expected relative latency (full network = 1.0)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"{title}  ({len(reports)} seeds; thick = mean)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
