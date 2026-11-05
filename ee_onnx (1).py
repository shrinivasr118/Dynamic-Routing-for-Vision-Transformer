"""
ee_onnx.py - REAL latency on the edge-CPU simulation (ONNX Runtime, 2 threads, batch 1).

The network is exported as segments (stem..exit1, exit1..exit2, ..., last..classifier). At inference the next segment
only runs if the router did not fire - so skipped work is genuinely skipped, exactly like your notebook ONNX cells.
Works for any router: free routers are scored in numpy from the logits, learned gates are exported with the segment.

    from ee_onnx import profile_onnx
    out = profile_onnx(res.model, res.teacher, test_loader, router="vib_gate", thresholds=[0.85, 0.85], n_samples=2000)
"""
from __future__ import annotations
import os, time
import numpy as np
import torch
import torch.nn as nn
from ee_core import np_free_score, FREE_ROUTERS


class _Seg(nn.Module):
    def __init__(self, model, k, router):
        super().__init__()
        self.model, self.k, self.router = model, k, router
        ex = model.exit_points
        self.last = k == model.K
        self.lo = 0 if k == 0 else ex[k - 1] + 1
        self.hi = model.adapter.n_units - 1 if self.last else ex[k]

    def forward(self, x):
        a = self.model.adapter
        if self.k == 0: x = a.start(x)
        for i in range(self.lo, self.hi + 1): x = a.step(i, x)
        if self.last: return a.finish(x)
        lg, g = self.model._exit(self.k, x, self.router)
        return (x, lg) if g is None else (x, lg, g)


class _Full(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)


def _export(m, arg, path, outs):
    kw = dict(input_names=["x"], output_names=outs)
    try: torch.onnx.export(m, arg, path, dynamo=False, **kw)      # legacy exporter (what your notebooks used)
    except TypeError: torch.onnx.export(m, arg, path, **kw)


def export_segments(model, router, out_dir, teacher=None):
    import onnxruntime as ort
    os.makedirs(out_dir, exist_ok=True)
    model = model.cpu().eval()
    segs = [_Seg(model, k, router).eval() for k in range(model.K + 1)]
    with torch.no_grad():
        d = torch.randn(1, 3, model.img_size, model.img_size); ins = [d]
        for s in segs[:-1]: ins.append(s(ins[-1])[0])
    has_gate = router in model.gates
    for k, s in enumerate(segs):
        outs = ["logits"] if s.last else (["feat", "logits", "gate"] if has_gate else ["feat", "logits"])
        _export(s, ins[k], f"{out_dir}/{router}_s{k + 1}.onnx", outs)
    base_path = None
    if teacher is not None:
        _export(_Full(teacher.cpu().eval()).eval(), d, f"{out_dir}/baseline.onnx", ["logits"])
        base_path = f"{out_dir}/baseline.onnx"
    return [f"{out_dir}/{router}_s{k + 1}.onnx" for k in range(model.K + 1)], base_path


def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _stats(t, correct, n, fracs=None):
    t = np.asarray(t)
    d = dict(acc=100.0 * correct / n, mean_ms=float(t.mean()), p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)))
    if fracs is not None: d["exit_pct"] = [100.0 * f / n for f in fracs]
    return d


def profile_onnx(model, teacher, loader, router, thresholds, n_samples=10000, threads=2, out_dir="./onnx_ee", export=True):
    """Returns dict(baseline=..., early_exit=..., latency_change_pct=...). Same images, same loop, same timer for both."""
    if export:
        seg_paths, base_path = export_segments(model, router, out_dir, teacher)
    else:
        seg_paths = [f"{out_dir}/{router}_s{k + 1}.onnx" for k in range(model.K + 1)]; base_path = f"{out_dir}/baseline.onnx"
    sess = [_session(p, threads) for p in seg_paths]
    sbase = _session(base_path, threads)
    samples = []
    for x, y in loader:
        for i in range(x.size(0)):
            samples.append((x[i:i + 1].numpy(), int(y[i])))
        if len(samples) >= n_samples: break
    samples = samples[:n_samples]
    for img, _ in samples[:10]:
        sbase.run(None, {"x": img}); sess[0].run(None, {"x": img})

    tb, cb = [], 0
    for img, lbl in samples:
        t0 = time.perf_counter(); lg = sbase.run(None, {"x": img})[0]; tb.append((time.perf_counter() - t0) * 1000)
        cb += int(lg.argmax(1)[0] == lbl)

    te, ce, cnt = [], 0, [0] * (model.K + 1)
    gate_router = router in model.gates
    for img, lbl in samples:
        t0 = time.perf_counter(); feat, stage, pred = img, None, None
        for k in range(model.K):
            out = sess[k].run(None, {"x": feat})
            feat, lg = out[0], out[1]
            s = float(1.0 / (1.0 + np.exp(-out[2].reshape(-1)[0]))) if gate_router else float(np_free_score(router, lg)[0])
            if s >= thresholds[k]:
                pred, stage = int(lg.argmax(1)[0]), k; break
        if stage is None:
            lg = sess[-1].run(None, {"x": feat})[0]; pred, stage = int(lg.argmax(1)[0]), model.K
        te.append((time.perf_counter() - t0) * 1000); cnt[stage] += 1; ce += int(pred == lbl)

    n = len(samples)
    base, ee = _stats(tb, cb, n), _stats(te, ce, n, cnt)
    return dict(baseline=base, early_exit=ee, router=router, thresholds=list(thresholds), n_samples=n, threads=threads,
               latency_change_pct=100.0 * (ee["mean_ms"] - base["mean_ms"]) / base["mean_ms"],
               acc_change=ee["acc"] - base["acc"])
