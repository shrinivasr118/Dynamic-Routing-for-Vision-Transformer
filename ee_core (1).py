"""
ee_core.py - building blocks for early-exit routing studies.

Backbones   : DeiT / ViT (timm, token models)  and  MobileViT (timm, conv + transformer stages)
Exit heads  : MLP head on [CLS ; mean(patch tokens)]  (ViT)   |   depthwise-conv head (MobileViT)
Routers     : msp, margin, entropy      training-free confidence scores from the exit head
              conf_gate                 BCE-trained gate on the 3 confidence features only (no latent)
              vib_latent                VIB latent only (no confidence features)
              vib_gate                  VIB latent + confidence features   (your final method)

All learned gates take DETACHED inputs, so they never change the backbone or the exit heads.
That means ONE training run produces identical heads for every router => a fair router comparison.

Everything here is a direct port of the working code in your DeiT-Tiny / MobileViT notebooks.
"""
from __future__ import annotations
import contextlib, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FREE_ROUTERS = ("msp", "margin", "entropy")
LEARNED_ROUTERS = ("conf_gate", "vib_latent", "vib_gate")
ALL_ROUTERS = FREE_ROUTERS + LEARNED_ROUTERS


# ----------------------------------------------------------------------------- confidence statistics
def confidence_feats(logits):
    """[max-prob, top1-top2 margin, 1 - normalised entropy]  (label-free, from the exit head only)."""
    p = F.softmax(logits, dim=-1)
    top2 = p.topk(2, dim=-1).values
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(-1) / math.log(p.size(-1))
    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], 1.0 - ent], dim=1)


def free_score(name, logits):
    """Training-free router score (higher = more confident = exit)."""
    f = confidence_feats(logits.detach().float())
    return {"msp": f[:, 0], "margin": f[:, 1], "entropy": f[:, 2]}[name]


def np_free_score(name, logits):
    """Numpy twin of free_score for one sample / batch of logits (used by ONNX profiling)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
    srt = np.sort(p, axis=-1)
    if name == "msp":
        return srt[..., -1]
    if name == "margin":
        return srt[..., -1] - srt[..., -2]
    ent = -(p * np.log(np.clip(p, 1e-8, None))).sum(-1) / math.log(p.shape[-1])
    return 1.0 - ent


def _fp32(t):
    return torch.autocast("cuda", enabled=False) if t.is_cuda else contextlib.nullcontext()


# ----------------------------------------------------------------------------- learned gates
class ConfGate(nn.Module):
    """BCE-trained gate on the 3 confidence features only. sigmoid(logit) ~ P(exit head is correct)."""
    def __init__(self, feat_dim=None, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat, logits):
        with _fp32(logits):
            g = self.net(confidence_feats(logits.detach().float())).squeeze(1)
        return g, g.new_zeros(())


class VIBGate(nn.Module):
    """Variational information bottleneck gate (Alemi et al. 2017): z ~ N(mu, sigma^2), KL(q(z|x) || N(0, I)).
    use_conf=True  -> gate sees [z ; confidence features]   (vib_gate)
    use_conf=False -> gate sees z only                       (vib_latent)"""
    def __init__(self, feat_dim, latent_dim=16, hidden=64, use_conf=True):
        super().__init__()
        self.use_conf = use_conf
        self.enc = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim + (3 if use_conf else 0), 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, feat, logits):
        with _fp32(feat):
            h = self.enc(feat.detach().float())
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h).clamp(-8.0, 8.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()
            inp = torch.cat([z, confidence_feats(logits.detach().float())], dim=1) if self.use_conf else z
            g = self.gate(inp).squeeze(1)
        return g, kl


def _make_gate(name, feat_dim, latent_dim):
    if name == "conf_gate":
        return ConfGate(feat_dim)
    if name == "vib_latent":
        return VIBGate(feat_dim, latent_dim, use_conf=False)
    if name == "vib_gate":
        return VIBGate(feat_dim, latent_dim, use_conf=True)
    raise ValueError(f"unknown learned router '{name}' (choose from {LEARNED_ROUTERS})")


# ----------------------------------------------------------------------------- exit heads
class ViTExitHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_classes))

    def forward(self, f):
        return self.net(f)


class ConvExitHead(nn.Module):
    """depthwise 3x3 -> 1x1 -> GAP -> Linear (a bare GAP+Linear reaches only ~25-35% on CIFAR-100 at MobileViT stage 2/3)."""
    def __init__(self, c, num_classes, width=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, width, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(inplace=True))
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.body(x), 1).flatten(1))


# ----------------------------------------------------------------------------- backbone adapters
class ViTAdapter:
    """timm DeiT / ViT. A 'unit' is one transformer block."""
    kind = "vit"
    head_is_feat = True          # exit head and router read the same pooled vector

    def __init__(self, base, exit_points=None, img_size=224):
        self.b = base
        self.n_units = len(base.blocks)
        n = self.n_units
        self.exit_points = list(exit_points) if exit_points is not None else [n // 3 - 1, 2 * n // 3 - 1]
        self.feat_dims = [2 * base.embed_dim] * len(self.exit_points)

    def start(self, x):
        b = self.b
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        if hasattr(b, "patch_drop"): x = b.patch_drop(x)
        if hasattr(b, "norm_pre"): x = b.norm_pre(x)
        return x

    def step(self, i, x):
        return self.b.blocks[i](x)

    def head_input(self, x):                       # CLS token + mean of patch tokens
        return torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)

    def router_feat(self, x):
        return self.head_input(x)

    def make_head(self, j, num_classes):
        return ViTExitHead(self.feat_dims[j], num_classes)

    def finish(self, x):
        return self.b.forward_head(self.b.norm(x))


class MobileViTAdapter:
    """timm MobileViT. A 'unit' is one entry of base.stages."""
    kind = "mobilevit"
    head_is_feat = False         # head reads the feature map, router reads its global-average-pool

    def __init__(self, base, exit_points=None, img_size=256):
        self.b = base
        self.n_units = len(base.stages)
        self.exit_points = list(exit_points) if exit_points is not None else [2, 3]
        was_training = base.training
        base.eval()                                  # probing in train mode would corrupt BatchNorm running stats
        dims = []
        with torch.no_grad():
            dev = next(base.parameters()).device
            x = base.stem(torch.randn(1, 3, img_size, img_size, device=dev))
            for i, st in enumerate(base.stages):
                x = st(x)
                if i in self.exit_points:
                    dims.append(x.shape[1])
        base.train(was_training)
        self.feat_dims = dims

    def start(self, x):
        return self.b.stem(x)

    def step(self, i, x):
        return self.b.stages[i](x)

    def head_input(self, x):
        return x

    def router_feat(self, x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def make_head(self, j, num_classes):
        return ConvExitHead(self.feat_dims[j], num_classes)

    def finish(self, x):
        b = self.b
        if getattr(b, "final_conv", None) is not None:
            x = b.final_conv(x)
        return b.forward_head(x)


def make_adapter(base, exit_points=None, img_size=224, kind=None):
    if kind is None:
        if hasattr(base, "blocks") and hasattr(base, "patch_embed"): kind = "vit"
        elif hasattr(base, "stages") and hasattr(base, "stem"): kind = "mobilevit"
        else: raise ValueError("Unrecognised backbone. Pass kind='vit' or 'mobilevit', or write an adapter (see ViTAdapter).")
    cls = {"vit": ViTAdapter, "mobilevit": MobileViTAdapter}[kind]
    ad = cls(base, exit_points, img_size)
    assert all(0 <= e < ad.n_units - 1 for e in ad.exit_points), "exit points must be before the last unit"
    assert ad.exit_points == sorted(ad.exit_points)
    return ad


# ----------------------------------------------------------------------------- the model
class EarlyExitNet(nn.Module):
    """Backbone + exit heads + learned gates (all learned routers share the same exit heads)."""
    def __init__(self, base_model, exit_points=None, num_classes=100, img_size=224,
                 routers=LEARNED_ROUTERS, latent_dim=16, kind=None):
        super().__init__()
        self.base_model = base_model
        self.adapter = make_adapter(base_model, exit_points, img_size, kind)
        self.exit_points = list(self.adapter.exit_points)
        self.K = len(self.exit_points)
        self.num_classes, self.img_size, self.latent_dim = num_classes, img_size, latent_dim
        self.exit_heads = nn.ModuleList([self.adapter.make_head(j, num_classes) for j in range(self.K)])
        self.gates = nn.ModuleDict()
        for r in routers:
            self.gates[r] = nn.ModuleList([_make_gate(r, self.adapter.feat_dims[j], latent_dim) for j in range(self.K)])

    def meta(self):
        return dict(exit_points=self.exit_points, num_classes=self.num_classes, img_size=self.img_size,
                    routers=list(self.gates.keys()), latent_dim=self.latent_dim, kind=self.adapter.kind)

    # -- one exit: head logits (+ gate logit of the requested learned router)
    def _exit(self, j, x, router=None):
        a = self.adapter
        h = a.head_input(x)
        lg = self.exit_heads[j](h)
        g = None
        if router in self.gates:
            f = h if a.head_is_feat else a.router_feat(x)
            g, _ = self.gates[router][j](f, lg)
        return lg, g

    def forward_train(self, x):
        """Batched pass through every exit. Returns (logits [K+1], gate_logits {router: [K]}, kl {router})."""
        a = self.adapter
        x = a.start(x)
        logits, gates = [], {r: [] for r in self.gates}
        kls = {r: x.new_zeros(()).float() for r in self.gates}
        j = 0
        for i in range(a.n_units):
            x = a.step(i, x)
            if j < self.K and i == self.exit_points[j]:
                h = a.head_input(x)
                lg = self.exit_heads[j](h)
                f = h if a.head_is_feat else a.router_feat(x)
                logits.append(lg)
                for r, g in self.gates.items():
                    gl, kl = g[j](f, lg)
                    gates[r].append(gl)
                    kls[r] = kls[r] + kl
                j += 1
        logits.append(a.finish(x))
        return logits, gates, kls

    @torch.no_grad()
    def forward_dynamic(self, x, router, thresholds):
        """TRUE early exit for ONE image: later units are never executed once the router fires."""
        assert not self.training and x.size(0) == 1, "forward_dynamic needs eval() and batch size 1"
        a = self.adapter
        x = a.start(x)
        j = 0
        for i in range(a.n_units):
            x = a.step(i, x)
            if j < self.K and i == self.exit_points[j]:
                lg, g = self._exit(j, x, router)
                s = torch.sigmoid(g.float()).item() if g is not None else free_score(router, lg).item()
                if s >= thresholds[j]:
                    return lg, j + 1
                j += 1
        return a.finish(x), self.K + 1


# ----------------------------------------------------------------------------- loss
class EarlyExitLoss(nn.Module):
    """total = weighted mean_k[(1-a)*CE_k + a*T^2*KD_k]
             + sum_routers [ lambda_gate * mean_k BCE(gate_k, head_k_is_correct) + beta*KL + lambda_hw*E[rel. latency] ]
    * Gate inputs are detached, so the gate terms only train the gates; head/backbone loss is router-independent.
    * Latency is in RELATIVE units (full network = 1.0), so lambda_hw does not depend on the hardware."""
    def __init__(self, costs, kd_temp=3.0, kd_alpha=0.5, beta=1e-4, lambda_gate=1.0, lambda_hw=0.05, final_weight=2.0):
        super().__init__()
        self.register_buffer("costs", torch.tensor(costs, dtype=torch.float32))
        self.kd_temp, self.kd_alpha = kd_temp, kd_alpha
        self.beta, self.lambda_gate, self.lambda_hw, self.final_weight = beta, lambda_gate, lambda_hw, final_weight

    def forward(self, logits, gates, kls, teacher_logits, y):
        T, a = self.kd_temp, self.kd_alpha
        soft_t = F.softmax(teacher_logits.float() / T, dim=-1)
        w = [1.0] * (len(logits) - 1) + [self.final_weight]
        head_loss = 0.0
        for wk, lg in zip(w, logits):
            lg = lg.float()
            ce = F.cross_entropy(lg, y)
            kd = F.kl_div(F.log_softmax(lg / T, dim=-1), soft_t, reduction="batchmean") * (T * T)
            head_loss = head_loss + wk * ((1 - a) * ce + a * kd)
        total = head_loss / sum(w)
        parts = {"head": float(total.detach())}
        correct = [(lg.detach().argmax(1) == y).float() for lg in logits[:-1]]
        for r, gl in gates.items():
            bce = sum(F.binary_cross_entropy_with_logits(g.float(), t) for g, t in zip(gl, correct)) / max(1, len(gl))
            reach, exp_lat = 1.0, 0.0
            for k, g in enumerate(gl):
                p = torch.sigmoid(g.float())
                exp_lat = exp_lat + reach * p * self.costs[k]
                reach = reach * (1.0 - p)
            exp_lat = (exp_lat + reach * self.costs[-1]).mean()
            total = total + self.lambda_gate * bce + self.beta * kls[r] + self.lambda_hw * exp_lat
            parts[f"{r}_bce"] = float(bce)
        return total, parts


# ----------------------------------------------------------------------------- hardware stage costs
@torch.no_grad()
def _median_ms(fn, dev, n=100, warm=20):
    for _ in range(warm): fn()
    if dev.type == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


@torch.no_grad()
def measure_stage_costs(model, dev, router=None, n=100, warm=20):
    """Median batch-1 time to reach exit k (including all earlier heads/gates and this exit's head/gate),
    divided by the time of the full network. Returns (relative_costs [K+1], ms [K+1])."""
    model = model.to(dev).eval()
    a = model.adapter
    x0 = torch.randn(1, 3, model.img_size, model.img_size, device=dev)

    def run(stop):
        def f():
            x = a.start(x0); j = 0
            for i in range(a.n_units):
                x = a.step(i, x)
                if j < model.K and i == model.exit_points[j]:
                    model._exit(j, x, router)
                    if stop == j: return
                    j += 1
            a.finish(x)
        return f

    ms = [_median_ms(run(j), dev, n, warm) for j in range(model.K)] + [_median_ms(run(None), dev, n, warm)]
    return [m / ms[-1] for m in ms], ms
