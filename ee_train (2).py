"""
ee_train.py - training, collecting and threshold selection for early-exit models.

Typical use (see run_example.py):
    from ee_train import train
    res = train(base_model, train_loader, val_loader, tag="deit_tiny", img_size=224, epochs=40)
    res.model          # best EarlyExitNet (loaded)
    res.costs          # {"cpu": [...], "gpu": [...]} relative stage costs measured on this machine

Any key in DEFAULTS can be overridden by keyword: train(..., lr_heads=1e-3, exit_points=[2, 3]).
"""
from __future__ import annotations
import copy, itertools, json, os
import numpy as np
import torch
import torch.optim as optim

from ee_core import (EarlyExitNet, EarlyExitLoss, ALL_ROUTERS, FREE_ROUTERS, LEARNED_ROUTERS,
                     free_score, measure_stage_costs)

DEFAULTS = dict(
    # --- model
    exit_points=None,            # None -> ViT: [3, 7] for 12 blocks | MobileViT: [2, 3]
    num_classes=100, img_size=224, latent_dim=16,
    routers=LEARNED_ROUTERS,     # learned gates trained together (free routers need no training)
    # --- optimisation
    epochs=30, lr_backbone=1e-5, lr_heads=2e-3, lr_router=1e-3, weight_decay=1e-4, clip=1.0, amp=True, seed=42,
    # --- loss
    beta=1e-4, lambda_gate=1.0, lambda_hw=0.05, hw_warmup=5, final_weight=2.0, kd_temp=3.0, kd_alpha=0.5,
    # --- threshold selection / checkpointing
    max_drop=1.0,                # allowed val-accuracy drop vs the baseline (points) when picking the best epoch
    select_routers=FREE_ROUTERS, # best epoch is judged ONLY by these routers. Free routers do not depend on the learned gates
                                 # (or on lambda_hw), so every run/seed/lambda picks its epoch the same neutral way.
    grid_mode="quantile", grid_n=20,
    # --- hardware cost model
    cost_target="cpu",           # which hardware's stage costs enter the training loss: "cpu" or "gpu"
    cpu_threads=2,               # CPU timing uses this many threads (your ONNX edge setting)
    out_dir="./ee_outputs",
)

FIXED_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]


# ----------------------------------------------------------------------------- evaluation helpers
@torch.no_grad()
def model_accuracy(model, loader, device):
    """Plain accuracy (%) of any nn.Module returning logits (use for the baseline / teacher)."""
    model.eval(); c = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item(); n += len(y)
    return 100.0 * c / n


@torch.no_grad()
def collect(model, loader, device):
    """One batched fp32 pass. Returns dict(P=[N,K+1] per-head predictions, Y=[N], S={router: [N,K] scores}).
    Scores: sigmoid(gate logit) for learned routers; msp / margin / (1-entropy) for free routers.
    In eval mode the VIB gate uses z = mu, identical to forward_dynamic."""
    model.eval()
    P, Y = [], []
    S = {r: [] for r in list(FREE_ROUTERS) + list(model.gates.keys())}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, gates, _ = model.forward_train(x)
        P.append(torch.stack([l.argmax(1) for l in logits], 1).cpu())
        for r in FREE_ROUTERS:
            S[r].append(torch.stack([free_score(r, l) for l in logits[:-1]], 1).cpu())
        for r in model.gates:
            S[r].append(torch.stack([torch.sigmoid(g.float()) for g in gates[r]], 1).cpu())
        Y.append(y)
    return dict(P=torch.cat(P).numpy(), Y=torch.cat(Y).numpy(),
                S={r: torch.cat(v).numpy() for r, v in S.items()})


# ----------------------------------------------------------------------------- threshold machinery (numpy)
def simulate(P, S_r, Y, thr, costs):
    """Per-sample routing simulation. Exit at the first exit j with score >= thr[j], else the final head.
    Returns (acc %, exit fractions [K+1], expected relative latency)."""
    N, K1 = P.shape
    K = K1 - 1
    stage = np.full(N, K)
    undecided = np.ones(N, dtype=bool)
    for j in range(K):
        take = undecided & (S_r[:, j] >= thr[j])
        stage[take] = j
        undecided &= ~take
    pred = P[np.arange(N), stage]
    fr = np.bincount(stage, minlength=K1) / N
    return float((pred == Y).mean() * 100.0), fr, float((fr * np.asarray(costs)).sum())


def make_grids(S_r, mode="quantile", n=20):
    """Threshold candidates per exit. 'quantile' spreads candidates over the router's own score distribution,
    so routers with different score scales get equally dense Pareto curves. inf = never exit here."""
    grids = []
    for j in range(S_r.shape[1]):
        g = np.quantile(S_r[:, j], np.linspace(0.0, 0.98, n)) if mode == "quantile" else np.array(FIXED_GRID)
        grids.append(np.unique(np.concatenate([g, [np.inf]])))
    return grids


def operating_points(P, S_r, Y, costs, grids):
    """All threshold combinations -> list of (thr tuple, acc, exit fractions, rel latency)."""
    return [(thr,) + simulate(P, S_r, Y, thr, costs) for thr in itertools.product(*grids)]


def best_under_budget(points, ref_acc, max_drop):
    """Lowest expected latency with acc >= ref_acc - max_drop (ties -> higher acc).
    If nothing is feasible, falls back to the never-exit point (= the final head alone)."""
    feas = [p for p in points if p[1] >= ref_acc - max_drop]
    if feas:
        return min(feas, key=lambda p: (p[3], -p[1])), True
    return max((p for p in points if np.all(np.isinf(p[0]))), key=lambda p: p[1]), False


def epoch_selection(data, costs, ref_acc, max_drop, grid_mode, grid_n, only=None):
    """Checkpoint criterion: mean over the routers in `only` of 'best latency within the accuracy budget'.
    Per-router results are recorded for ALL routers (for logging) but only `only` decides the epoch."""
    per, scores = {}, []
    for r, S_r in data["S"].items():
        pts = operating_points(data["P"], S_r, data["Y"], costs, make_grids(S_r, grid_mode, grid_n))
        (thr, acc, fr, lat), ok = best_under_budget(pts, ref_acc, max_drop)
        per[r] = dict(thr=[float(t) for t in thr], acc=acc, lat=lat, ok=ok)
        if only is None or r in only:
            scores.append(-lat * 100 + acc * 1e-3 if ok else -1e6 + acc)
    return float(np.mean(scores)), per


# ----------------------------------------------------------------------------- cost measurement
def measure_costs(model, targets=("cpu", "gpu"), cpu_threads=2, router=None):
    """Relative stage costs on the requested hardware(s). Works on a copy; restores the thread count."""
    out = {}
    for t in targets:
        m = copy.deepcopy(model).eval()
        if t == "gpu":
            if torch.cuda.is_available():
                out["gpu"] = measure_stage_costs(m, torch.device("cuda"), router)[0]
        else:
            prev = torch.get_num_threads()
            torch.set_num_threads(cpu_threads)
            try:
                out["cpu"] = measure_stage_costs(m.cpu(), torch.device("cpu"), router)[0]
            finally:
                torch.set_num_threads(prev)
        del m
    return out


class EEResult:
    """Everything train() produces."""
    def __init__(self, **kw): self.__dict__.update(kw)


# ----------------------------------------------------------------------------- training
def train(base_model, train_loader, val_loader, teacher=None, device=None, tag="ee",
          costs=None, verbose=True, **overrides):
    """Fine-tune a trained backbone with exit heads + gates (knowledge distillation from `teacher`).

    base_model : trained timm model (weights already loaded). It is copied, never modified.
    teacher    : frozen model used for distillation + reference accuracy. Default: a frozen copy of base_model.
    costs      : optional list of relative stage costs [K+1]; default = measured on cfg['cost_target'] hardware.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; valid keys: {sorted(DEFAULTS)}")
    c = {**DEFAULTS, **overrides}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(c["amp"]) and device.type == "cuda"
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    os.makedirs(c["out_dir"], exist_ok=True)

    if teacher is None:
        teacher = copy.deepcopy(base_model)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False
    ref_val = model_accuracy(teacher, val_loader, device)

    model = EarlyExitNet(copy.deepcopy(base_model).to(device), c["exit_points"], c["num_classes"], c["img_size"],
                         c["routers"], c["latent_dim"]).to(device)

    all_costs = measure_costs(model, ("cpu", "gpu"), c["cpu_threads"], router=(list(c["routers"])[-1] if c["routers"] else None))
    if costs is None:
        key = c["cost_target"] if c["cost_target"] in all_costs else next(iter(all_costs))
        costs = all_costs[key]
    model.to(device)
    if verbose:
        print(f"[{tag}] baseline val acc {ref_val:.2f}% | exits {model.exit_points} | "
              f"stage costs used {[round(v, 3) for v in costs]} | measured {{ {', '.join(k + ': ' + str([round(v, 3) for v in x]) for k, x in all_costs.items())} }}")

    groups = [{"params": model.base_model.parameters(), "lr": c["lr_backbone"], "weight_decay": c["weight_decay"]},
              {"params": model.exit_heads.parameters(), "lr": c["lr_heads"], "weight_decay": c["weight_decay"]}]
    if len(model.gates):
        groups.append({"params": model.gates.parameters(), "lr": c["lr_router"], "weight_decay": c["weight_decay"]})
    opt = optim.AdamW(groups)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
    crit = EarlyExitLoss(costs, c["kd_temp"], c["kd_alpha"], c["beta"], c["lambda_gate"], c["lambda_hw"], c["final_weight"]).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bb_params = list(model.base_model.parameters()) + list(model.exit_heads.parameters())
    gate_params = list(model.gates.parameters())

    best_path, last_path = f"{c['out_dir']}/{tag}_best.pth", f"{c['out_dir']}/{tag}_last.pth"
    best_score, best_epoch, history = -1e18, 0, []

    for epoch in range(c["epochs"]):
        model.train()
        crit.lambda_hw = c["lambda_hw"] * min(1.0, epoch / max(1, c["hw_warmup"]))
        run, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                t_logits = teacher(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, gates, kls = model.forward_train(x)
            loss, _ = crit(logits, gates, kls, t_logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(bb_params, c["clip"])          # clipped separately so gates cannot change
            if gate_params: torch.nn.utils.clip_grad_norm_(gate_params, c["clip"])   # how the heads are trained
            scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        sched.step()

        data = collect(model, val_loader, device)
        head_acc = [float((data["P"][:, k] == data["Y"]).mean() * 100) for k in range(data["P"].shape[1])]
        score, per = epoch_selection(data, costs, ref_val, c["max_drop"], c["grid_mode"], c["grid_n"], only=c["select_routers"])
        history.append(dict(epoch=epoch + 1, loss=run / max(1, nb), head_acc=head_acc, per_router=per, score=score))
        if verbose:
            s = " | ".join(f"{r} {v['acc']:.1f}/{v['lat']:.3f}" for r, v in per.items())
            print(f"[{tag}] ep {epoch + 1:02d}/{c['epochs']} loss {run / max(1, nb):.3f} | head acc "
                  + "/".join(f"{a:.1f}" for a in head_acc) + f" | acc/relLat per router: {s}")
        torch.save(dict(model=model.state_dict(), meta=model.meta(), cfg=c, costs=all_costs, epoch=epoch + 1), last_path)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            torch.save(dict(model=model.state_dict(), meta=model.meta(), cfg=c, costs=all_costs, epoch=epoch + 1), best_path)

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    if verbose: print(f"[{tag}] best epoch {best_epoch} -> {best_path}")
    with open(f"{c['out_dir']}/{tag}_history.json", "w") as f:
        json.dump(dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}, ref_val_acc=ref_val,
                       costs=all_costs, history=history), f, indent=1, default=str)
    return EEResult(model=model, teacher=teacher, cfg=c, costs=all_costs, train_costs=costs, history=history,
                    best_epoch=best_epoch, ckpt_path=best_path, ref_val_acc=ref_val, device=device, tag=tag)


def load_early_exit(base_model, ckpt_path, device=None):
    """Rebuild a trained EarlyExitNet. `base_model` = same timm architecture (weights get overwritten)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = EarlyExitNet(base_model, **ck["meta"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck
