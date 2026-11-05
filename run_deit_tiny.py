"""
run_deit_tiny.py - DeiT-Tiny / CIFAR-100: train all routers, lambda_hw sweep, Pareto plots, optional real ONNX latency.

    python run_deit_tiny.py --smoke                      # 2-3 min pipeline check on random data (do this first)
    python run_deit_tiny.py                              # full run: 40 epochs for each lambda_hw in --lambdas
    python run_deit_tiny.py --lambdas 0.05               # a single run
    python run_deit_tiny.py --reuse                      # skip training for lambdas that already have a checkpoint
    python run_deit_tiny.py --onnx                       # + real edge-CPU latency at one operating point per router

Put this file next to ee_core.py / ee_train.py / ee_analysis.py / ee_onnx.py.
Outputs (under --out): lhw<value>/ per-lambda checkpoints, history, router table (csv/json), pareto png;
                       summary_*.csv, pareto_all_lambdas.png, onnx_latency.json
"""
import argparse, csv, json, os, sys
import numpy as np
import torch, timm
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ee_core import ALL_ROUTERS
from ee_train import train, load_early_exit, model_accuracy, EEResult
from ee_analysis import full_report

# DeiT-Tiny recipe taken from your notebook (vib_* config cell)
RECIPE = dict(exit_points=[3, 7], img_size=224, num_classes=100, lr_backbone=2e-5, lr_heads=1e-3, lr_router=1e-3,
              beta=1e-4, latent_dim=16, lambda_gate=1.0, hw_warmup=5, final_weight=2.0, max_drop=0.5, cost_target="cpu")
BUDGETS = (0.25, 0.5, 1.0, 2.0)


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--baseline", default="./phase1_outputs/deit_tiny_cifar100_best.pth")
    a.add_argument("--data", default="./data")
    a.add_argument("--out", default="./ee_outputs/deit_tiny")
    a.add_argument("--epochs", type=int, default=40)
    a.add_argument("--lambdas", type=float, nargs="+", default=[0.05,0.1,0.12, 0.15], help="lambda_hw values, one full training run each")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) - 1)))
    a.add_argument("--reuse", action="store_true", help="load an existing best checkpoint instead of retraining")
    a.add_argument("--onnx", action="store_true", help="measure real ONNX edge-CPU latency")
    a.add_argument("--onnx_lambda", default="auto", help="'auto' = lambda with lowest VALIDATION latency for vib_gate, or a number")
    a.add_argument("--onnx_budget", type=float, default=0.5)
    a.add_argument("--onnx_samples", type=int, default=2000)
    a.add_argument("--smoke", action="store_true", help="random data, tiny images, 1 epoch: checks the pipeline only")
    return a.parse_args()


def get_loaders(args):
    if args.smoke:
        def ld(n, shuffle):
            return DataLoader(TensorDataset(torch.randn(n, 3, 64, 64), torch.randint(0, 100, (n,))), batch_size=32, shuffle=shuffle)
        return ld(128, True), ld(64, False), ld(64, False)
    import torchvision, torchvision.transforms as T
    M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_t = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=9),
                         T.ToTensor(), T.Normalize(M, S), T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)])
    test_t = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(M, S)])
    full_aug = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=train_t)
    full_clean = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=test_t)
    test_ds = torchvision.datasets.CIFAR100(args.data, train=False, download=True, transform=test_t)
    idx = torch.randperm(len(full_aug), generator=torch.Generator().manual_seed(42)).tolist()     # same 45k/5k split as your notebook
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    return (DataLoader(Subset(full_aug, idx[:45000]), shuffle=True, persistent_workers=args.workers > 0, **kw),
            DataLoader(Subset(full_clean, idx[45000:]), shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


def reuse_result(base, ckpt, tag, device):
    model, ck = load_early_exit(base, ckpt, device)
    teacher = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100, **({"img_size": 64} if ARGS.smoke else {}))
    if not ARGS.smoke:
        teacher.load_state_dict(torch.load(ARGS.baseline, map_location="cpu"))
    teacher = teacher.to(device).eval()
    costs = ck["costs"]; key = ck["cfg"]["cost_target"]
    return EEResult(model=model, teacher=teacher, cfg=ck["cfg"], costs=costs, train_costs=costs.get(key, next(iter(costs.values()))),
                    history=[], best_epoch=ck.get("epoch", -1), ckpt_path=ckpt, ref_val_acc=None, device=device, tag=tag)


def summary(reports, lams, out):
    os.makedirs(out, exist_ok=True)
    lines, csv_rows = [], []
    for b in (0.5, 1.0):
        lines.append(f"\n=== budget {b} points: test acc / latency saving %  (thresholds picked on validation) ===")
        lines.append(f"{'router':<12}" + "".join(f"{'lhw=' + str(l):>20}" for l in lams))
        for r in ALL_ROUTERS:
            cells = []
            for l in lams:
                row = next((x for x in reports[l]["rows"] if x["router"] == r and x["budget"] == b), None)
                cells.append("n/a" if row is None else f"{row['test_acc']:.2f} / {row['saving_pct']:.1f}%")
                if row: csv_rows.append([b, r, l, row["test_acc"], row["test_drop"], row["saving_pct"], row["oracle_capture_pct"], row["feasible"]])
            lines.append(f"{r:<12}" + "".join(f"{c:>20}" for c in cells))
    o = reports[lams[0]]["oracle"]["oracle_iso"]
    lines.append(f"\noracle (iso-accuracy): saving {100 * (1 - o['lat']):.1f}% at baseline accuracy {reports[lams[0]]['ref_test']:.2f}%")
    txt = "\n".join(lines); print(txt)
    with open(f"{out}/summary.txt", "w") as f: f.write(txt)
    with open(f"{out}/summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["budget", "router", "lambda_hw", "test_acc", "test_drop", "saving_pct", "oracle_capture_pct", "feasible"]); w.writerows(csv_rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    for ax, r in zip(axs.ravel(), ALL_ROUTERS):
        for l in lams:
            rows = sorted(reports[l]["curves"].get(r, []), key=lambda d: d["rel_lat"])
            ax.plot([d["rel_lat"] for d in rows], [d["test_acc"] for d in rows], marker="o", ms=3, label=f"lambda_hw={l}")
        rep = reports[lams[0]]
        ax.scatter([1.0], [rep["ref_test"]], c="k", marker="s", zorder=5)
        ax.scatter([rep["oracle"]["oracle_iso"]["lat"]], [rep["oracle"]["oracle_iso"]["acc"]], c="gold", edgecolors="k", marker="*", s=150, zorder=6)
        ax.set_title(r); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8); fig.supxlabel("expected relative latency (full network = 1.0)  | black square = baseline, star = iso-accuracy oracle"); fig.supylabel("test accuracy (%)")
    fig.tight_layout(); fig.savefig(f"{out}/pareto_all_lambdas.png", dpi=200); plt.close(fig)


def main():
    global ARGS
    ARGS = args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(args)
    kw = {"img_size": 64} if args.smoke else {}
    base = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100, **kw)
    if not args.smoke:
        base.load_state_dict(torch.load(args.baseline, map_location="cpu"))
    b_val, b_test = model_accuracy(base.to(device), val_loader, device), model_accuracy(base, test_loader, device)
    print(f"baseline  val {b_val:.2f}%  test {b_test:.2f}%   (no flip-TTA here, so ~0.2-0.5 below the TTA number in your notebook)")
    base = base.cpu()

    recipe = dict(RECIPE)
    if args.smoke: recipe["img_size"] = 64
    reports, results = {}, {}
    for lam in args.lambdas:
        tag, out_l = f"deit_tiny_lhw{lam}", f"{args.out}/lhw{lam}"
        ckpt = f"{out_l}/{tag}_best.pth"
        if args.reuse and os.path.exists(ckpt):
            print(f"\n##### lambda_hw={lam}: reusing {ckpt}")
            res = reuse_result(timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=100, **kw), ckpt, tag, device)
        else:
            print(f"\n##### lambda_hw={lam}: training {args.epochs} epochs")
            res = train(base, train_loader, val_loader, device=device, tag=tag, out_dir=out_l,
                        epochs=1 if args.smoke else args.epochs, lambda_hw=lam, seed=args.seed, grid_n=8 if args.smoke else 20, **recipe)
        reports[lam], results[lam] = full_report(res, val_loader, test_loader, out_dir=out_l, budgets=BUDGETS), res
    summary(reports, args.lambdas, args.out)

    if args.onnx:
        from ee_onnx import profile_onnx
        if args.onnx_lambda == "auto":      # chosen on VALIDATION latency, never on test
            def val_lat(l):
                r = next(x for x in reports[l]["rows"] if x["router"] == "vib_gate" and x["budget"] == args.onnx_budget)
                return r["val_rel_lat"]
            lam = min(args.lambdas, key=val_lat)
        else:
            lam = float(args.onnx_lambda)
        res, out = results[lam], {}
        print(f"\n##### ONNX edge-CPU latency | lambda_hw={lam} | budget {args.onnx_budget} | {args.onnx_samples} samples")
        for r in ALL_ROUTERS:
            row = next(x for x in reports[lam]["rows"] if x["router"] == r and x["budget"] == args.onnx_budget)
            o = profile_onnx(res.model, res.teacher, test_loader, r, row["thr"], n_samples=args.onnx_samples,
                             out_dir=f"{args.out}/onnx_lhw{lam}", export=True)
            out[r] = {**o, "thresholds": row["thr"]}
            e, b = o["early_exit"], o["baseline"]
            print(f"{r:<12} acc {e['acc']:.2f} (base {b['acc']:.2f}) | mean {e['mean_ms']:.2f} ms vs {b['mean_ms']:.2f} ({o['latency_change_pct']:+.1f}%) | p95 {e['p95_ms']:.2f} | exits% {[round(v, 1) for v in e['exit_pct']]}")
        with open(f"{args.out}/onnx_latency.json", "w") as f: json.dump(out, f, indent=1, default=str)
    print(f"\nDone. Everything is in {args.out}")


if __name__ == "__main__":
    main()
