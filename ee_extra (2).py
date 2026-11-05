"""
ee_extra.py - tests of the signals proposed by Dib (2026, Knowledge-Based Systems) inside YOUR matched-accuracy protocol.

Adds, on top of the trained early-exit models (no retraining):
  * entropy trajectory   dH_k = H_k - H_{k-1}                      (cross-exit signal)
  * cross-exit agreement 1[pred_k == pred_{k-1}]                    (patience-style signal)
  * Mahalanobis distance of the exit feature to the nearest class mean (shared covariance, fitted on clean train images)
  * logistic-regression routers over these signals with Dib-style staged conditional training
  * supermartingale test: P(H_{k+1} <= H_k | head k correct)
  * conformal exit thresholds (error among exited samples <= alpha, finite-sample corrected)
  * oracle-match rate: how often the router picks the oracle's exit
Protocol: validation is split in two halves. Half A fits the logistic routers, half B picks thresholds (for ALL routers, so the
comparison is fair), test scores them. References are the model's OWN final head (val-B / test), so every drop is "vs final head".
"""
from __future__ import annotations
import numpy as np
import torch

from ee_core import confidence_feats
from ee_analysis import select_and_eval, latency_at_accuracy

LR_KINDS = {"lr_conf": "", "lr_traj": "traj", "lr_agree": "agree", "lr_maha": "maha", "lr_all": "traj agree maha"}


# ----------------------------------------------------------------------------- forward helper
def _forward_all(model, x):
    """Logits and router features at every exit (+ final logits), plus sigmoid scores of the trained gates."""
    a = model.adapter
    x = a.start(x)
    logits, feats, gates = [], [], {r: [] for r in model.gates}
    j = 0
    for i in range(a.n_units):
        x = a.step(i, x)
        if j < model.K and i == model.exit_points[j]:
            h = a.head_input(x)
            lg = model.exit_heads[j](h)
            f = h if a.head_is_feat else a.router_feat(x)
            for r, g in model.gates.items():
                gates[r].append(torch.sigmoid(g[j](f, lg)[0].float()))
            logits.append(lg.float()); feats.append(f.float()); j += 1
    logits.append(a.finish(x).float())
    return logits, feats, gates


# ----------------------------------------------------------------------------- Mahalanobis statistics
@torch.no_grad()
def fit_class_stats(model, loader, device, max_n=10000, num_classes=100):
    """Class means + shared covariance of the router feature at every exit (clean, un-augmented training images)."""
    model.eval()
    S1 = S2 = None
    cnt = torch.zeros(num_classes, dtype=torch.float64, device=device)
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, feats, _ = _forward_all(model, x)
        if S1 is None:
            S1 = [torch.zeros(num_classes, f.shape[1], dtype=torch.float64, device=device) for f in feats]
            S2 = [torch.zeros(f.shape[1], f.shape[1], dtype=torch.float64, device=device) for f in feats]
        for j, f in enumerate(feats):
            f = f.double()
            S1[j].index_add_(0, y, f); S2[j] += f.T @ f
        cnt += torch.bincount(y, minlength=num_classes).double()
        seen += len(y)
        if seen >= max_n: break
    stats = []
    n = cnt.sum()
    for j in range(len(S1)):
        mu = S1[j] / cnt.clamp_min(1).unsqueeze(1)
        cov = (S2[j] - (mu * cnt.unsqueeze(1)).T @ mu) / n
        cov = cov + 1e-3 * torch.eye(cov.shape[0], dtype=torch.float64, device=device) * cov.diagonal().mean()
        P = torch.linalg.inv(cov)
        Pmu = mu @ P
        stats.append(dict(mu=mu.float(), P=P.float(), Pmu=Pmu.float(), muPmu=(mu * Pmu).sum(1).float(), seen=(cnt > 0)))
    return stats


def _maha(f, st):
    d = (f @ st["P"] * f).sum(1, keepdim=True) - 2 * f @ st["Pmu"].T + st["muPmu"].unsqueeze(0)
    d = d.masked_fill(~st["seen"].unsqueeze(0), float("inf"))
    return d.min(1).values.clamp_min(0)


# ----------------------------------------------------------------------------- signal collection
@torch.no_grad()
def collect_signals(model, loader, device, stats=None):
    """Per-sample signals for every exit AND the final head. H = normalised entropy, M = margin, MSP = max prob,
    D = Mahalanobis distance (exits only), G = trained-gate scores."""
    model.eval()
    acc = {k: [] for k in ("Y", "P", "H", "M", "MSP", "D")}
    G = {r: [] for r in model.gates}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, feats, gates = _forward_all(model, x)
        cf = [confidence_feats(l) for l in logits]
        acc["P"].append(torch.stack([l.argmax(1) for l in logits], 1).cpu())
        acc["MSP"].append(torch.stack([c[:, 0] for c in cf], 1).cpu())
        acc["M"].append(torch.stack([c[:, 1] for c in cf], 1).cpu())
        acc["H"].append(torch.stack([1 - c[:, 2] for c in cf], 1).cpu())
        acc["D"].append(torch.stack([_maha(f, stats[j]) if stats else torch.zeros(len(y), device=device)
                                     for j, f in enumerate(feats)], 1).cpu())
        for r in G: G[r].append(torch.stack(gates[r], 1).cpu())
        acc["Y"].append(y)
    out = {k: torch.cat(v).numpy() for k, v in acc.items()}
    out["G"] = {r: torch.cat(v).numpy() for r, v in G.items()}
    return out


def _sub(sig, idx):
    o = {k: v[idx] for k, v in sig.items() if k != "G"}
    o["G"] = {r: v[idx] for r, v in sig["G"].items()}
    return o


# ----------------------------------------------------------------------------- supermartingale test
def supermartingale(sig):
    """For each transition k -> k+1: among samples whose head k is correct (and wrong), P(H_{k+1} <= H_k) and mean entropies."""
    rows = []
    corr = sig["P"] == sig["Y"][:, None]
    for k in range(sig["P"].shape[1] - 1):
        for name, m in (("correct", corr[:, k]), ("wrong", ~corr[:, k])):
            if m.sum() == 0: continue
            h0, h1 = sig["H"][m, k], sig["H"][m, k + 1]
            rows.append(dict(transition=f"E{k + 1}->E{k + 2 if k + 2 <= sig['P'].shape[1] - 1 else 'final'}", head_k=name,
                             n=int(m.sum()), p_decrease=float((h1 <= h0).mean()), mean_H_k=float(h0.mean()), mean_H_next=float(h1.mean())))
    return rows


# ----------------------------------------------------------------------------- logistic routers (Dib-style, staged)
def _features(sig, j, kind):
    f = [sig["MSP"][:, j], sig["M"][:, j], 1 - sig["H"][:, j]]
    if "traj" in kind: f.append(sig["H"][:, j] - sig["H"][:, j - 1] if j > 0 else np.zeros(len(sig["Y"])))
    if "agree" in kind: f.append((sig["P"][:, j] == sig["P"][:, j - 1]).astype(float) if j > 0 else np.zeros(len(sig["Y"])))
    if "maha" in kind: f.append(np.log1p(sig["D"][:, j]))
    return np.stack(f, 1)


def _fit_logistic(X, y, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    if y.min() == y.max():                                   # degenerate (all correct / all wrong)
        return dict(mu=mu, sd=sd, w=np.zeros(X.shape[1]), b=10.0 if y[0] else -10.0)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w[:-1] ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    w = w.detach().numpy()
    return dict(mu=mu, sd=sd, w=w[:-1], b=float(w[-1]))


def _predict(m, X):
    return 1.0 / (1.0 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


def fit_lr_router(sigA, kind, staged=True, keep=0.6):
    """One logistic regression per exit predicting 'head k is correct'. Staged: exit k>0 is fitted only on the `keep` fraction of
    samples that the previous exit's predictor finds hardest (these are the ones that actually reach exit k)."""
    K = sigA["P"].shape[1] - 1
    models, prev = [], None
    for j in range(K):
        idx = np.arange(len(sigA["Y"]))
        if staged and j > 0:
            idx = np.argsort(prev)[:max(20, int(keep * len(prev)))]
        X, y = _features(sigA, j, kind), (sigA["P"][:, j] == sigA["Y"])
        m = _fit_logistic(X[idx], y[idx].astype(float))
        models.append(m); prev = _predict(m, X)
    return models


def predict_lr_router(models, sig):
    return np.stack([_predict(m, _features(sig, j, kind=models[0]["kind"])) for j, m in enumerate(models)], 1)


# ----------------------------------------------------------------------------- oracle match + conformal
def _stages(S, thr):
    N, K = S.shape
    st, und = np.full(N, K), np.ones(N, bool)
    for j in range(K):
        take = und & (S[:, j] >= thr[j]); st[take] = j; und &= ~take
    return st


def oracle_match(P, Y, S, thr):
    """Fraction of samples that leave at exactly the oracle's exit (earliest correct head if the final head is also correct)."""
    N, K1 = P.shape; K = K1 - 1
    corr = P == Y[:, None]
    first = np.where(corr[:, :K].any(1), corr[:, :K].argmax(1), K)
    orc = np.where(corr[:, K], first, K)
    return float((_stages(S, thr) == orc).mean() * 100)


def conformal_thresholds(S, P, Y, alpha):
    """Per exit, on the samples that reach it: smallest threshold with (errors_exited + 1) / (n_exited + 1) <= alpha
    (conformal risk control for a loss bounded by 1; guarantees E[error | exits here] <= alpha under exchangeability)."""
    K = S.shape[1]; thr, und = [], np.ones(len(Y), bool)
    for j in range(K):
        idx = np.where(und)[0]
        order = idx[np.argsort(-S[idx, j])]
        err = np.cumsum(P[order, j] != Y[order]); n = np.arange(1, len(order) + 1)
        ok = np.where((err + 1) / (n + 1) <= alpha)[0]
        t = S[order[ok.max()], j] if len(ok) else np.inf
        thr.append(float(t)); und &= ~(S[:, j] >= t)
    return thr


def conformal_report(valB, test, names, costs, alphas=(0.02, 0.05, 0.10)):
    rows = []
    K = valB["P"].shape[1] - 1
    for r in names:
        for a in alphas:
            thr = conformal_thresholds(valB["S"][r], valB["P"], valB["Y"], a)
            st = _stages(test["S"][r], thr); N = len(st)
            errs = [float((test["P"][st == j, j] != test["Y"][st == j]).mean()) if (st == j).any() else float("nan") for j in range(K)]
            fr = np.bincount(st, minlength=K + 1) / N
            acc = float((test["P"][np.arange(N), st] == test["Y"]).mean() * 100)
            rows.append(dict(router=r, alpha=a, thr=thr, exit_frac=fr.tolist(), exit_err=errs, test_acc=acc,
                             rel_lat=float((fr * np.asarray(costs)).sum()), held=bool(all(np.isnan(e) or e <= a for e in errs))))
    return rows


# ----------------------------------------------------------------------------- full per-model evaluation
def evaluate_extra(model, val_loader, test_loader, train_loader_clean, device, costs, num_classes=100, maha_n=10000,
                   budgets=(0.5, 1.0, 2.0), dense=None, staged=True, keep=0.6, split_seed=0):
    dense = np.round(np.linspace(0.0, 5.0, 26), 3) if dense is None else dense
    stats = fit_class_stats(model, train_loader_clean, device, maha_n, num_classes)
    sv, st = collect_signals(model, val_loader, device, stats), collect_signals(model, test_loader, device, stats)
    perm = np.random.RandomState(split_seed).permutation(len(sv["Y"])); A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]
    sA, sB = _sub(sv, A), _sub(sv, B)

    def scores(sig):
        s = {"msp": sig["MSP"][:, :-1], "margin": sig["M"][:, :-1], "entropy": 1 - sig["H"][:, :-1]}
        s.update({r: g for r, g in sig["G"].items()})
        return s
    SB, ST = scores(sB), scores(st)
    for name, kind in LR_KINDS.items():
        ms = fit_lr_router(sA, kind, staged, keep)
        for m in ms: m["kind"] = kind
        SB[name], ST[name] = predict_lr_router(ms, sB), predict_lr_router(ms, st)
    names = list(SB)
    valB = dict(P=sB["P"], Y=sB["Y"], S=SB)
    test = dict(P=st["P"], Y=st["Y"], S=ST)
    ref_val = float((sB["P"][:, -1] == sB["Y"]).mean() * 100); ref_test = float((st["P"][:, -1] == st["Y"]).mean() * 100)
    rows = select_and_eval(valB, test, costs, ref_val, ref_test, budgets, routers=names)
    for r in rows:
        r["oracle_match_pct"] = oracle_match(test["P"], test["Y"], test["S"][r["router"]], r["thr"]) if r["thr"] else float("nan")
    curves = {}
    for row in select_and_eval(valB, test, costs, ref_val, ref_test, dense, routers=names):
        if not row["router"].startswith("oracle"): curves.setdefault(row["router"], []).append(row)
    return dict(rows=rows, curves=curves, test=test, ref_test=ref_test, ref_val=ref_val, names=names, supermartingale=supermartingale(st),
                conformal=conformal_report(valB, test, [n for n in ("margin", "conf_gate", "lr_all") if n in names], costs))


def matched_table(reports, drops=(0.5, 1.0), ref="margin"):
    """Same as ee_analysis.matched_accuracy_table but for any router names; reference = each model's own final head."""
    names = [n for n in reports[0]["names"]]
    out = []
    for d in drops:
        sav = {r: np.array([100.0 * (1.0 - latency_at_accuracy(rep["curves"][r], rep["ref_test"] - d)) for rep in reports]) for r in names}
        for r in names:
            v, diff = sav[r], sav[r] - sav[ref]
            out.append(dict(drop=d, router=r, saving_mean=float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan"),
                            saving_std=float(np.nanstd(v)) if np.any(~np.isnan(v)) else float("nan"),
                            diff_vs_ref_mean=float(np.nanmean(diff)) if np.any(~np.isnan(diff)) else float("nan"),
                            diff_vs_ref_std=float(np.nanstd(diff)) if np.any(~np.isnan(diff)) else float("nan"),
                            wins=int(np.nansum(diff > 0)), n=len(reports), ref=ref, valid=int(np.sum(~np.isnan(v)))))
    return out


# ----------------------------------------------------------------------------- paired bootstrap over test images
def bootstrap_prepare(report, costs):
    """Per router and dense budget: correctness and cost of every test image at the validation-selected thresholds."""
    t = report["test"]; P, Y = t["P"], t["Y"]; N = len(Y)
    prep = {}
    for r, rows in report["curves"].items():
        rows = sorted(rows, key=lambda d: d["budget"])
        corr = np.zeros((len(rows), N), np.float32); cost = np.zeros_like(corr)
        for i, row in enumerate(rows):
            st = _stages(t["S"][r], row["thr"])
            corr[i] = (P[np.arange(N), st] == Y); cost[i] = np.asarray(costs)[st]
        prep[r] = (corr, cost)
    return prep, (P[:, -1] == Y).astype(np.float32)


def bootstrap_matched(preps, final_corrs, drops=(1.0, 2.0), B=500, ref="margin", seed=0):
    """Resample test images (same resample for every seed and router); recompute each router's matched-accuracy saving on the
    resampled set. Returns {(drop, router): array of savings per resample} (mean over seeds). Captures TEST-SAMPLING noise only;
    training-seed noise is in the seed tables."""
    N = len(final_corrs[0]); rng = np.random.RandomState(seed)
    routers = list(preps[0]); sav = {(d, r): [] for d in drops for r in routers}
    with np.errstate(all="ignore"):
        for _ in range(B):
            idx = rng.randint(0, N, N)
            per = {(d, r): [] for d in drops for r in routers}
            for prep, fc in zip(preps, final_corrs):
                ref_acc = 100 * fc[idx].mean()
                for r in routers:
                    corr, cost = prep[r]
                    acc, lat = 100 * corr[:, idx].mean(1), cost[:, idx].mean(1)
                    rows = [dict(rel_lat=float(l), test_acc=float(a)) for l, a in zip(lat, acc)]
                    for d in drops: per[(d, r)].append(100.0 * (1.0 - latency_at_accuracy(rows, ref_acc - d)))
            for k, v in per.items(): sav[k].append(np.nanmean(v) if np.any(~np.isnan(v)) else np.nan)
    return {k: np.array(v) for k, v in sav.items()}, ref


def format_bootstrap(sav, ref, equiv=0.5):
    lines = []
    for d in sorted({k[0] for k in sav}):
        lines.append(f"\n=== paired bootstrap over test images: saving(router) - saving({ref}) at <= {d} pt drop; 95% CI; equivalence margin +-{equiv} ===")
        lines.append(f"{'router':<12}{'mean diff':>10}{'95% CI':>22}{'P(|diff|<=' + str(equiv) + ')':>16}   verdict")
        for r in [k[1] for k in sav if k[0] == d]:
            diff = sav[(d, r)] - sav[(d, ref)]
            diff = diff[~np.isnan(diff)]
            if r == ref or len(diff) == 0: continue
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = "equivalent" if (lo > -equiv and hi < equiv) else ("better" if lo > 0 else ("worse" if hi < 0 else "inconclusive"))
            ci = "[" + format(lo, "+.2f") + ", " + format(hi, "+.2f") + "]"
            lines.append(f"{r:<12}{diff.mean():>+10.2f}{ci:>22}{(np.abs(diff) <= equiv).mean():>16.2f}   {verdict}")
    return "\n".join(lines)
