"""
train_baseline_in100.py - phase 1 for ImageNet-100: the full (no-exit) baseline of each backbone.

Starts from the timm ImageNet-1k pretrained weights, keeps the 100 matching rows of the 1000-way classifier, fine-tunes a few epochs.
NOTE for the paper: the backbones were pretrained on ImageNet-1k, which contains these training images.

    python train_baseline_in100.py --model deit_tiny     --root /path/to/imagenet
    python train_baseline_in100.py --model mobilevit_xs  --root /path/to/imagenet --epochs 8
    python train_baseline_in100.py --model deit_base --root /path/to/imagenet --weights_path ./deit_base_1k.pth    # offline server

Output: ./phase1_in100_outputs/<model>_in100_best.pth (plain state_dict, the default baseline path of run_ee.py --dataset imagenet100) + in100_classes.txt
"""
import argparse, os, sys
import numpy as np
import torch, timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ee
from data_in100 import get_in100_loaders, all_wnids, select_wnids
from ee_train import model_accuracy


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True, choices=list(run_ee.MODELS))
    a.add_argument("--root", default=r"./", help="ImageNet root containing train/ and val/ class folders")
    a.add_argument("--classes_file", default=None); a.add_argument("--out", default="./phase1_in100_outputs")
    a.add_argument("--epochs", type=int, default=5); a.add_argument("--batch", type=int, default=None)
    a.add_argument("--lr", type=float, default=5e-5, help="backbone lr"); a.add_argument("--head_lr", type=float, default=5e-4)
    a.add_argument("--wd", type=float, default=0.05); a.add_argument("--label_smoothing", type=float, default=0.1)
    a.add_argument("--workers", type=int, default=8); a.add_argument("--seed", type=int, default=42)
    a.add_argument("--weights_path", default=None, help="local 1000-class state_dict (if the server cannot download timm weights)")
    a.add_argument("--no_pretrained", action="store_true", help="random init (only for pipeline tests)")
    a.add_argument("--val_per_class", type=int, default=50)
    args = a.parse_args()
    cfg = run_ee.MODELS[args.model]; img = cfg["img"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); use_amp = device.type == "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed); os.makedirs(args.out, exist_ok=True)

    train_dir = os.path.join(args.root, "train")
    all_w, sel = all_wnids(train_dir), select_wnids(train_dir, args.classes_file)
    open(f"{args.out}/in100_classes.txt", "w").write("\n".join(sel))
    train_loader, val_loader, test_loader = get_in100_loaders(args.root, cfg["timm"], img, args.batch or cfg["batch"], args.workers,
                                                              args.classes_file, None, args.val_per_class)
    kw = {}
    have_weights = (not args.no_pretrained) or args.weights_path is not None
    model = timm.create_model(cfg["timm"], pretrained=(not args.no_pretrained and args.weights_path is None), num_classes=1000, **kw)
    if args.weights_path: model.load_state_dict(torch.load(args.weights_path, map_location="cpu"))
    clf = model.get_classifier(); W, B = clf.weight.data.clone(), clf.bias.data.clone()
    model.reset_classifier(100); new = model.get_classifier()
    idx = [all_w.index(w) for w in sel] if len(all_w) == 1000 else None     # timm's 1000-way order = sorted wnids
    if have_weights and idx is not None and W.shape[0] == 1000:
        new.weight.data.copy_(W[idx]); new.bias.data.copy_(B[idx]); print("[head] initialised from the matching 100 rows of the pretrained 1000-way classifier")
    else:
        print("[head] WARNING: random head (needs the full 1000-class train folder + pretrained weights to reuse the classifier rows); train longer")
    model = model.to(device)
    print(f"[start] val acc before fine-tuning: {model_accuracy(model, val_loader, device):.2f}%")

    head_ids = {id(p) for p in new.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
                             {"params": list(new.parameters()), "lr": args.head_lr}], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, path = -1.0, f"{args.out}/{args.model}_in100_best.pth"
    for ep in range(args.epochs):
        model.train(); run = n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); run += loss.item(); n += 1
        sched.step()
        v = model_accuracy(model, val_loader, device)
        print(f"ep {ep + 1}/{args.epochs} loss {run / max(1, n):.3f} val acc {v:.2f}%")
        if v > best: best = v; torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"best val {best:.2f}% | TEST acc {model_accuracy(model, test_loader, device):.2f}% | saved {path}")


if __name__ == "__main__":
    main()
