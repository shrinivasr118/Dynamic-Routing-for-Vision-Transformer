"""
run_extra.py - Dib-style signals on your trained models (needs the checkpoints from run_ee.py; nothing is retrained).

    python run_extra.py --model deit_tiny --lambdas 0.15 --seeds 42 43 44
    python run_extra.py --model mobilevit_xs --seeds 42 43 44 --maha_n 10000
    python run_extra.py --model deit_tiny --smoke --out ./smoke_out --lambdas 0.05 --seeds 42     # pipeline check

Output: <out>/<model>/extra/summary.txt (+ per-seed json).
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np
import torch, timm
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ee
from ee_train import load_early_exit
from ee_analysis import format_matched
from ee_extra import evaluate_extra, matched_table, bootstrap_prepare, bootstrap_matched, format_bootstrap


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True, choices=list(run_ee.MODELS))
    a.add_argument("--data", default="./"); a.add_argument("--out", default="./ee_outputs")
    a.add_argument("--lambdas", type=float, nargs="+", default=[0.15]); a.add_argument("--seeds", type=int, nargs="+", default=[42])
    a.add_argument("--maha_n", type=int, default=10000, help="clean train images used for class means/covariance")
    a.add_argument("--keep", type=float, default=0.6, help="staged training: fraction of hardest samples used for exit k>0")
    a.add_argument("--no_staged", action="store_true")
    a.add_argument("--workers", type=int, default=4); a.add_argument("--batch", type=int, default=None)
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--bootstrap", type=int, default=0, help="B test-set resamples for paired CIs / equivalence (e.g. 500); 0 = off")
    a.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet100"])
    a.add_argument("--classes_file", default=None); a.add_argument("--subset_per_class", type=int, default=None)
    a.add_argument("--val_per_class", type=int, default=50)
    args = a.parse_args()
    cfg = run_ee.MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = SimpleNamespace(smoke=args.smoke, data=args.data, batch=args.batch, workers=args.workers, dataset=args.dataset,
                         classes_file=args.classes_file, subset_per_class=args.subset_per_class, val_per_class=args.val_per_class)
    root = run_ee.out_root_for(args.out, args.model, args.dataset)
    train_loader, val_loader, test_loader = run_ee.get_loaders(ns, cfg)
    if getattr(train_loader, "clean_loader", None) is not None:
        clean = train_loader.clean_loader
    elif hasattr(train_loader.dataset, "indices") and hasattr(val_loader.dataset, "dataset"):   # clean (un-augmented) train subset
        clean = DataLoader(Subset(val_loader.dataset.dataset, train_loader.dataset.indices), batch_size=val_loader.batch_size,
                           shuffle=False, num_workers=args.workers)
    else:
        clean = val_loader
    mk = {"img_size": 64} if (args.smoke and cfg["timm"].startswith("deit")) else {}
    out_dir = f"{root}/extra"; os.makedirs(out_dir, exist_ok=True)

    for lam in args.lambdas:
        reports, costs_list = [], []
        for seed in args.seeds:
            tag = f"{args.model}_lhw{lam}_s{seed}"
            ckpt = f"{root}/lhw{lam}/seed{seed}/{tag}_best.pth"
            model, ck = load_early_exit(timm.create_model(cfg["timm"], pretrained=False, num_classes=100, **mk), ckpt, device)
            costs = ck["costs"].get("cpu") or next(iter(ck["costs"].values()))
            print(f"[{args.model} lambda {lam} seed {seed}] evaluating ...")
            rep = evaluate_extra(model, val_loader, test_loader, clean, device, costs, maha_n=args.maha_n,
                                 staged=not args.no_staged, keep=args.keep)
            reports.append(rep); costs_list.append(costs)
            json.dump({k: v for k, v in rep.items() if k != "curves"}, open(f"{out_dir}/seed{seed}_lhw{lam}.json", "w"), indent=1, default=str)

        lines = [f"{args.model} | lambda_hw {lam} | {len(reports)} seed(s) | savings vs each model's OWN final head | thresholds on val-half-B"]
        lines.append(format_matched(matched_table(reports, drops=(0.5, 1.0, 2.0))))
        lines.append("\n=== oracle-match % and conformal (alpha = target error among exited samples), seed-averaged ===")
        for r in reports[0]["names"]:
            om = [x["oracle_match_pct"] for rep in reports for x in rep["rows"] if x["router"] == r and x["budget"] == 1.0]
            lines.append(f"{r:<12} oracle-match@1.0pt {np.nanmean(om):5.1f}%")
        for al in (0.02, 0.05, 0.10):
            for r in ("margin", "lr_all"):
                cs = [x for rep in reports for x in rep["conformal"] if x["router"] == r and x["alpha"] == al]
                if cs:
                    lines.append(f"conformal alpha={al:<5} {r:<8} test acc {np.mean([c['test_acc'] for c in cs]):.2f}  rel.lat {np.mean([c['rel_lat'] for c in cs]):.3f}"
                                 f"  guarantee held in {sum(c['held'] for c in cs)}/{len(cs)} seeds  exit err {[round(float(np.nanmean([c['exit_err'][j] for c in cs])), 3) for j in range(len(cs[0]['exit_err']))]}")
        lines.append("\n=== entropy supermartingale: P(H_next <= H_k | head k correct / wrong), seed 1 ===")
        for s in reports[0]["supermartingale"]:
            lines.append(f"{s['transition']:<10} head {s['head_k']:<8} n={s['n']:<6} P(dec)={s['p_decrease']:.3f}  meanH {s['mean_H_k']:.3f} -> {s['mean_H_next']:.3f}")
        if args.bootstrap > 0:
            print(f"[bootstrap] {args.bootstrap} resamples x {len(reports)} seed(s) ...")
            prepped = [bootstrap_prepare(rep, c) for rep, c in zip(reports, costs_list)]
            sav, ref = bootstrap_matched([q[0] for q in prepped], [q[1] for q in prepped], drops=(1.0, 2.0), B=args.bootstrap)
            lines.append(format_bootstrap(sav, ref, equiv=0.5))
        txt = "\n".join(lines); print("\n" + "=" * 100 + "\n" + txt)
        open(f"{out_dir}/summary_lhw{lam}.txt", "w").write(txt)


if __name__ == "__main__":
    main()
