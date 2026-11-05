"""
run_ee.py - one script for every backbone: DeiT-Tiny / Small / Base and MobileViT-XS on CIFAR-100.
For each seed it trains the exit heads + gates once (all six routers share the heads), then compares the routers
at matched accuracy against the oracle, and aggregates over seeds.

    python run_ee.py --model deit_tiny  --smoke                 # 3-minute pipeline check on random data. Do this first.
    python run_ee.py --model deit_tiny                          # lambda_hw sweep 0.05 0.1 0.12 0.15, seed 42, 40 epochs each
    python run_ee.py --model deit_tiny --lambdas 0.15 --seeds 42 43 44    # noise study: one lambda, 3 seeds
    python run_ee.py --model mobilevit_xs                       # same recipe study on the hybrid
    python run_ee.py --model deit_base --seeds 42 --epochs 30   # one seed, fewer epochs
    python run_ee.py --model deit_tiny --reuse                  # skip seeds that already have a checkpoint
    python run_ee.py --model deit_tiny --onnx                   # + real edge-CPU latency (first seed only)

Needs ee_core.py, ee_train.py, ee_analysis.py (and ee_onnx.py for --onnx) in the same folder.
Output (under --out/<model>/): lhw<value>/seed<k>/ (checkpoint, history, router table, pareto png),
lhw<value>/matched_accuracy.txt + router_table_mean_std.csv + pareto_seeds.png, lambda_summary.txt, pareto_all_lambdas.png, onnx_latency.json
"""
import argparse, csv, json, os, sys
import numpy as np
import torch, timm
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ee_core import ALL_ROUTERS
from ee_train import train, load_early_exit, model_accuracy, EEResult
from ee_analysis import (full_report, aggregate_seeds, format_aggregate, matched_accuracy_table, format_matched, plot_seed_pareto)

# Per-backbone settings, copied from your notebooks. Edit freely.
MODELS = {
    "deit_tiny": dict(timm="deit_tiny_patch16_224", img=224, batch=256, aug="rrc",
                      baseline="./phase1_outputs/deit_tiny_cifar100_best.pth",
                      recipe=dict(exit_points=[3, 7], lr_backbone=2e-5, lr_heads=1e-3, lr_router=1e-3, final_weight=2.0)),
    "deit_small": dict(timm="deit_small_patch16_224", img=224, batch=256, aug="pad12",
                       baseline="./phase1_outputs/deit_small_cifar100_best.pth",
                       recipe=dict(exit_points=[3, 7], lr_backbone=2e-5, lr_heads=1e-3, lr_router=1e-3, final_weight=2.0)),
    "deit_base": dict(timm="deit_base_patch16_224", img=224, batch=128, aug="pad12",
                      baseline="./phase1_deit_base_outputs/deit_base_cifar100_best.pth",
                      recipe=dict(exit_points=[3, 7], lr_backbone=2e-5, lr_heads=1e-3, lr_router=1e-3, final_weight=2.0)),
    "mobilevit_xs": dict(timm="mobilevit_xs", img=256, batch=128, aug="pad16",
                         baseline="./phase1_mobilevit_outputs/mobilevit_xs_cifar100_best.pth",
                         recipe=dict(exit_points=[2, 3], lr_backbone=1e-5, lr_heads=2e-3, lr_router=1e-3, final_weight=1.0)),
}
BUDGETS = (0.25, 0.5, 1.0, 2.0)


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True, choices=list(MODELS))
    a.add_argument("--baseline", default=None, help="override the baseline checkpoint path")
    a.add_argument("--data", default="./")
    a.add_argument("--out", default="./ee_outputs")
    a.add_argument("--epochs", type=int, default=None, help="default: 40 for cifar100, 15 for imagenet100")
    a.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100"])
    a.add_argument("--classes_file", default=None, help="imagenet100: file with the 100 wnids (one per line)")
    a.add_argument("--subset_per_class", type=int, default=None, help="imagenet100: cap train images per class (faster runs)")
    a.add_argument("--val_per_class", type=int, default=50)
    a.add_argument("--seeds", type=int, nargs="+", default=[42], help="one training run per (lambda, seed). Use 42 43 44 for a noise estimate")
    a.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.1, 0.12, 0.15], help="lambda_hw values: one full run each (per seed)")
    a.add_argument("--max_drop", type=float, default=0.5, help="val accuracy budget used when picking the best epoch")
    a.add_argument("--batch", type=int, default=None)
    a.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) - 1)), help="0 if DataLoader workers crash (Windows)")
    a.add_argument("--reuse", action="store_true")
    a.add_argument("--onnx", action="store_true")
    a.add_argument("--onnx_budget", type=float, default=0.5)
    a.add_argument("--onnx_lambda", default="auto", help="'auto' = lambda with the lowest VALIDATION latency for vib_gate, or a number")
    a.add_argument("--onnx_samples", type=int, default=2000)
    a.add_argument("--onnx_routers", nargs="+", default=["msp", "margin", "conf_gate", "vib_gate"])
    a.add_argument("--smoke", action="store_true", help="random data, tiny images, 1 epoch, 1 seed: pipeline check only")
    return a.parse_args()


def out_root_for(out, model, dataset):
    return f"{out}/{model}" if dataset == "cifar100" else f"{out}/{dataset}/{model}"


def get_loaders(args, cfg):
    if getattr(args, "dataset", "cifar100") == "imagenet100" and not args.smoke:
        from data_in100 import get_in100_loaders
        return get_in100_loaders(args.data, cfg["timm"], cfg["img"], args.batch or cfg["batch"], args.workers,
                                 args.classes_file, args.subset_per_class, getattr(args, "val_per_class", 50))
    img = 64 if args.smoke else cfg["img"]
    if args.smoke:
        def ld(n, sh): return DataLoader(TensorDataset(torch.randn(n, 3, img, img), torch.randint(0, 100, (n,))), batch_size=32, shuffle=sh)
        return ld(128, True), ld(64, False), ld(64, False)
    import torchvision, torchvision.transforms as T
    M, S = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if cfg["aug"] == "rrc":      # DeiT-Tiny notebook
        geo = [T.RandomResizedCrop(img, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
        erase = T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    else:                        # DeiT-Base / MobileViT notebooks
        pad = 16 if cfg["aug"] == "pad16" else 12
        geo = [T.Resize((img, img)), T.RandomCrop(img, padding=pad, padding_mode="reflect"), T.RandomHorizontalFlip()]
        erase = T.RandomErasing(p=0.25, scale=(0.02, 0.33), value="random")
    train_t = T.Compose(geo + [T.RandAugment(num_ops=2, magnitude=9), T.ToTensor(), T.Normalize(M, S), erase])
    test_t = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(M, S)])
    full_aug = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=train_t)
    full_clean = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=test_t)
    test_ds = torchvision.datasets.CIFAR100(args.data, train=False, download=True, transform=test_t)
    idx = torch.randperm(len(full_aug), generator=torch.Generator().manual_seed(42)).tolist()     # fixed 45k/5k split (as in your notebooks)
    kw = dict(batch_size=args.batch or cfg["batch"], num_workers=args.workers, pin_memory=True)
    return (DataLoader(Subset(full_aug, idx[:45000]), shuffle=True, persistent_workers=args.workers > 0, drop_last=False, **kw),
            DataLoader(Subset(full_clean, idx[45000:]), shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


def main():
    args = get_args()
    cfg = MODELS[args.model]
    if args.epochs is None: args.epochs = 15 if args.dataset == "imagenet100" else 40
    baseline_path = args.baseline or (cfg["baseline"] if args.dataset == "cifar100" else f"./phase1_in100_outputs/{args.model}_in100_best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    out_root = out_root_for(args.out, args.model, args.dataset)
    os.makedirs(out_root, exist_ok=True)
    seeds, epochs = (args.seeds[:1], 1) if args.smoke else (args.seeds, args.epochs)
    img = 64 if args.smoke else cfg["img"]
    mk_kw = {"img_size": 64} if (args.smoke and cfg["timm"].startswith("deit")) else {}

    def new_model():
        return timm.create_model(cfg["timm"], pretrained=False, num_classes=100, **mk_kw)

    train_loader, val_loader, test_loader = get_loaders(args, cfg)
    base = new_model()
    if not args.smoke:
        base.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    bv, bt = model_accuracy(base.to(device), val_loader, device), model_accuracy(base, test_loader, device)
    base = base.cpu()
    print(f"[{args.model}] baseline val {bv:.2f}%  test {bt:.2f}%  (no flip-TTA)")

    seeds_run = seeds
    n_runs = len(args.lambdas) * len(seeds_run)
    print(f"-> {n_runs} training run(s): lambdas {args.lambdas} x seeds {seeds_run}, {epochs} epochs each")
    reports_by, results_by = {}, {}
    for lam in args.lambdas:
        reports_by[lam], results_by[lam] = [], {}
        for seed in seeds_run:
            tag, out_s = f"{args.model}_lhw{lam}_s{seed}", f"{out_root}/lhw{lam}/seed{seed}"
            ckpt = f"{out_s}/{tag}_best.pth"
            if args.reuse and os.path.exists(ckpt):
                print(f"\n##### lambda_hw={lam} seed {seed}: reusing {ckpt}")
                model, ck = load_early_exit(new_model(), ckpt, device)
                teacher = new_model()
                if not args.smoke: teacher.load_state_dict(torch.load(baseline_path, map_location="cpu"))
                costs, key = ck["costs"], ck["cfg"]["cost_target"]
                res = EEResult(model=model, teacher=teacher.to(device).eval(), cfg=ck["cfg"], costs=costs,
                               train_costs=costs.get(key, next(iter(costs.values()))), history=[], best_epoch=ck.get("epoch", -1),
                               ckpt_path=ckpt, ref_val_acc=None, device=device, tag=tag)
            else:
                print(f"\n##### lambda_hw={lam} seed {seed}: training {epochs} epochs")
                res = train(base, train_loader, val_loader, device=device, tag=tag, out_dir=out_s, epochs=epochs, seed=seed,
                            lambda_hw=lam, max_drop=args.max_drop, img_size=img, num_classes=100, hw_warmup=5,
                            cost_target="cpu", grid_n=8 if args.smoke else 20, **{k: v for k, v in cfg["recipe"].items()})
            results_by[lam][seed] = res
            reports_by[lam].append(full_report(res, val_loader, test_loader, out_dir=out_s, budgets=BUDGETS))

    # ---------------- aggregate over seeds, per lambda
    for lam, reports in reports_by.items():
        agg = aggregate_seeds([r["rows"] for r in reports])
        matched = matched_accuracy_table(reports, drops=(0.5, 1.0), ref="margin")
        o = np.mean([r["oracle"]["oracle_iso"]["lat"] for r in reports])
        txt = (f"{args.model} | lambda_hw {lam} | {len(reports)} seed(s) | baseline test acc {np.mean([r['ref_test'] for r in reports]):.2f}%\n"
               + format_aggregate(agg) + "\n" + format_matched(matched) + f"\n\noracle (iso-accuracy) saving: {100 * (1 - o):.1f}%")
        print("\n" + "=" * 100 + "\n" + txt)
        d = f"{out_root}/lhw{lam}"
        open(f"{d}/matched_accuracy.txt", "w").write(txt)
        with open(f"{d}/router_table_mean_std.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
        plot_seed_pareto(reports, f"{d}/pareto_seeds.png", f"{args.model} lambda_hw={lam}")

    # ---------------- all lambdas side by side (matched-accuracy saving, mean over seeds)
    lams = list(reports_by)
    lines = []
    for drop in (0.5, 1.0):
        lines.append(f"\n=== latency saving % at test accuracy within {drop} pt of baseline (mean over {len(seeds_run)} seed(s)) ===")
        lines.append(f"{'router':<12}" + "".join(f"{'lhw=' + str(l):>12}" for l in lams))
        tabs = {l: {(m['router']): m for m in matched_accuracy_table(reports_by[l], drops=(drop,), ref='margin')} for l in lams}
        for r in ALL_ROUTERS:
            lines.append(f"{r:<12}" + "".join(f"{tabs[l][r]['saving_mean']:>12.1f}" for l in lams))
    lines.append("\nSanity check: with the gate-independent epoch selection the free routers (msp, margin, entropy) should look")
    lines.append("almost identical across lambdas. Only the learned gates can depend on lambda_hw.")
    txt = "\n".join(lines); print(txt); open(f"{out_root}/lambda_summary.txt", "w").write(txt)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    for ax, r in zip(axs.ravel(), ALL_ROUTERS):
        for l in lams:
            rows_by_seed = [sorted(rep["curves"][r], key=lambda d: d["budget"]) for rep in reports_by[l]]
            lat = np.mean([[d["rel_lat"] for d in rw] for rw in rows_by_seed], axis=0)
            acc = np.mean([[d["test_acc"] for d in rw] for rw in rows_by_seed], axis=0)
            o_ = np.argsort(lat); ax.plot(lat[o_], acc[o_], marker="o", ms=3, label=f"lambda_hw={l}")
        rep0 = reports_by[lams[0]][0]
        ax.scatter([1.0], [rep0["ref_test"]], c="k", marker="s", zorder=5)
        ax.scatter([rep0["oracle"]["oracle_iso"]["lat"]], [rep0["oracle"]["oracle_iso"]["acc"]], c="gold", edgecolors="k", marker="*", s=150, zorder=6)
        ax.set_title(r); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8)
    fig.supxlabel("expected relative latency (full network = 1.0)  | black square = baseline, star = iso-accuracy oracle"); fig.supylabel("test accuracy (%)")
    fig.tight_layout(); fig.savefig(f"{out_root}/pareto_all_lambdas.png", dpi=200); plt.close(fig)

    # ---------------- optional: real ONNX latency (first seed; lambda picked on VALIDATION latency, never on test)
    if args.onnx:
        from ee_onnx import profile_onnx
        if args.onnx_lambda == "auto":
            lam = min(lams, key=lambda l: next(x for x in reports_by[l][0]["rows"] if x["router"] == "vib_gate" and x["budget"] == args.onnx_budget)["val_rel_lat"])
        else:
            lam = float(args.onnx_lambda)
        seed = seeds_run[0]; res, rep = results_by[lam][seed], reports_by[lam][0]
        print(f"\n##### ONNX edge-CPU | lambda_hw={lam} seed {seed} | budget {args.onnx_budget} | {args.onnx_samples} samples")
        out = {}
        for r in args.onnx_routers:
            row = next(x for x in rep["rows"] if x["router"] == r and x["budget"] == args.onnx_budget)
            o = profile_onnx(res.model, res.teacher, test_loader, r, row["thr"], n_samples=args.onnx_samples,
                             out_dir=f"{out_root}/onnx_lhw{lam}_seed{seed}")
            out[r] = {**o, "thresholds": row["thr"]}
            e, b = o["early_exit"], o["baseline"]
            print(f"{r:<12} acc {e['acc']:.2f} (base {b['acc']:.2f}) | mean {e['mean_ms']:.2f} ms vs {b['mean_ms']:.2f} ({o['latency_change_pct']:+.1f}%) | p95 {e['p95_ms']:.2f} | exits% {[round(v, 1) for v in e['exit_pct']]}")
        json.dump(out, open(f"{out_root}/onnx_latency.json", "w"), indent=1, default=str)
    print(f"\nDone. Results in {out_root}")


if __name__ == "__main__":
    main()
