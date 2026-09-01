"""Training loop for the multi-task RGB hand model (COMP0248 CW1 LSA).

Everything the coursework requires to be our own lives here: the loop, the loss composition,
the schedule, checkpointing and model selection. Losses are *composed* from PyTorch
primitives (``binary_cross_entropy_with_logits``, ``l1_loss``, ``cross_entropy``) — no
detection or segmentation library is imported anywhere in this project.

Two things in here are easy to get wrong and are therefore called out in code:

1. **Sparse mask supervision.** Only 14.1% of RealSense training frames carry a hand mask
   (2 keyframes per clip); every frame carries a clip-level gesture label. Detection and
   segmentation terms are averaged over the annotated subset of the batch only, while the
   classification term is averaged over the whole batch. A batch with zero annotated frames
   must contribute a classification gradient and nothing else — not a NaN.

2. **Target-free model selection.** The brief forbids using smartphone data in training, and
   selecting a checkpoint on the smartphone set would silently make Experiment 3's headline
   number meaningless. Selection uses held-out RealSense contributors, optionally scored
   through the held-out *pseudo-target* shift (``--select-on pseudo``). The real smartphone
   set is never read by this file.

Usage
-----
    python -m src.train --config configs/e1.yaml --out runs/e1
    python -m src.train --config configs/e3.yaml --out runs/e3 --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from src import utils                                    # noqa: E402
from src.model import HandNet, decode_detection          # noqa: E402
from src.dataloader import build_loaders                 # noqa: E402


# ======================================================================================
# configuration
# ======================================================================================
@dataclass
class Config:
    """Every knob of a run. Serialised verbatim into the checkpoint and the log."""
    # data
    index_path: str = ""            # packed index.json holding the train+val splits
    test_index_path: str = ""       # packed index.json for a --test pack (RealSense test / phone)
    img_size: tuple[int, int] = (384, 288)          # (W, H)
    batch_size: int = 48
    num_workers: int = 8

    # augmentation
    geometric: bool = True
    photometric: str = "jitter"                     # "none"|"jitter"|"cpr"|"randconv"|"aprs"|"augmix"
    cpr_stages: list[str] | None = None             # None = all stages (ablation block B)
    aug_strength: float = 1.0

    # model
    norm: str = "bn"                                # "bn"|"gn"|"ibn"
    width: float = 1.0
    use_mask_attn_pool: bool = True
    heads: tuple[str, ...] = ("det", "seg", "cls")

    # loss weights
    w_heat: float = 1.0
    w_size: float = 0.1
    w_off: float = 1.0
    w_giou: float = 1.0
    w_bce: float = 1.0
    w_dice: float = 1.0
    w_cls: float = 1.0
    label_smoothing: float = 0.05
    loss_weighting: str = "fixed"                   # "fixed" | "uncertainty"

    # optimisation
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    min_lr_factor: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999

    # bookkeeping
    seed: int = 0
    deterministic: bool = False
    select_on: str = "val"                          # "val" | "pseudo"
    select_metric: str = "combined"                 # "combined"|"macro_f1"|"seg_miou"|"det_acc50"
    eval_every: int = 1
    save_every: int = 1
    tag: str = ""

    def merged(self, overrides: dict) -> "Config":
        d = asdict(self)
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in d:
                raise KeyError(f"unknown config key: {k}")
            d[k] = v
        d["img_size"] = tuple(d["img_size"])
        d["heads"] = tuple(d["heads"])
        return Config(**d)


def _read_cfg_file(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:                              # keep the dependency optional
        return json.loads(text)


def load_config(path: str | None, overrides: dict) -> Config:
    """Layer ``defaults <- extends-chain <- this file <- CLI overrides``.

    A config may declare ``extends: base.yaml`` (resolved relative to its own directory). That
    is what makes "e1 and e3 differ in exactly one field" a property of the files rather than a
    claim about two full copies that someone must keep in sync by hand: e3.yaml contains one
    line of its own, so nothing else *can* drift between them.
    """
    base = Config()
    chain: list[dict] = []
    seen: set[Path] = set()
    cur = Path(path).resolve() if path else None
    while cur is not None:
        if cur in seen:
            raise ValueError(f"circular 'extends' chain at {cur}")
        seen.add(cur)
        raw = _read_cfg_file(cur)
        parent = raw.pop("extends", None)
        chain.append(raw)
        cur = (cur.parent / parent).resolve() if parent else None
    for raw in reversed(chain):                      # outermost ancestor first
        base = base.merged(raw)
    return base.merged(overrides)


# ======================================================================================
# loss
# ======================================================================================
def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss on a centre heatmap (CornerNet / CenterNet form).

    ``target`` is a Gaussian-splatted map whose peaks are exactly 1. Only those peaks are
    positives; every other cell is a negative whose loss is *down-weighted* by
    ``(1-target)**beta``, so cells near a true centre are barely penalised. A plain BCE here
    would fight the Gaussian's own skirt and blur the peak.

    Normalised by the number of positives, which is what makes the value comparable across
    batches with different numbers of annotated frames.
    """
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on the hand class, per image then averaged.

    Paired with BCE rather than used alone: BCE gives a well-behaved per-pixel gradient while
    Dice directly optimises the overlap ratio the metric reports, and is insensitive to the
    97:3 background:hand imbalance that would otherwise let BCE win by predicting background.
    """
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()


class UncertaintyWeights(nn.Module):
    """Learnable homoscedastic task weighting (Kendall, Gal & Cipolla, CVPR 2018).

    ``L = sum_i exp(-s_i) * L_i + s_i`` with ``s_i = log(sigma_i^2)`` learned. Off by default
    (``loss_weighting="fixed"``): on a dataset this small it is one more thing that can drift.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names
        self.log_var = nn.Parameter(torch.zeros(len(names)))

    def forward(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, n in enumerate(self.names):
            if n in terms:
                total = total + torch.exp(-self.log_var[i]) * terms[n] + self.log_var[i]
        return total


class MultiTaskLoss(nn.Module):
    """Composes the six loss terms and applies the sparse-mask rule."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.uw = UncertaintyWeights(["det", "seg", "cls"]) if cfg.loss_weighting == "uncertainty" else None

    def forward(self, out: dict, batch: dict, stride: int = 4) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        has = batch["has_mask"]
        n_ann = int(has.sum().item())
        parts: dict[str, torch.Tensor] = {}
        zero = torch.zeros((), device=has.device)

        # ---- classification: every frame, annotated or not -------------------------------
        if "cls" in out:
            parts["cls"] = self.ce(out["cls"], batch["label"])

        # ---- detection + segmentation: annotated frames only -----------------------------
        # A batch can legitimately contain zero annotated frames (14.1% annotation density).
        # Emitting a real zero here, rather than skipping the key, keeps the logged term
        # series continuous and keeps `total` a tensor with a graph in every branch.
        if n_ann == 0:
            for k in ("heat", "size", "off", "giou", "bce", "dice"):
                parts.setdefault(k, zero)
        else:
            idx = has.nonzero(as_tuple=True)[0]

            if "seg" in out:
                sl, sm = out["seg"][idx], batch["mask"][idx]
                parts["bce"] = F.binary_cross_entropy_with_logits(sl, sm)
                parts["dice"] = dice_loss(sl, sm)

            if "heat" in out:
                heat_t, size_t, off_t, ind, mask_pos = build_det_targets(
                    batch["box"][idx], out["heat"].shape[-2:], stride, has.device)
                parts["heat"] = focal_heatmap_loss(out["heat"][idx], heat_t)

                # size/offset are supervised at the single centre cell of each image; gather
                # there rather than masking a dense map, which would average in ~6900 cells of
                # meaningless target per image.
                size_p = gather_at(out["size"][idx], ind)          # (n,2) log w, log h
                off_p = gather_at(out["off"][idx], ind)            # (n,2)
                parts["size"] = F.l1_loss(size_p, size_t)
                parts["off"] = F.l1_loss(off_p, off_t)

                # GIoU on the decoded box: the two required detection metrics are IoU-based,
                # and L1 on log-size is one change of variables away from the thing we report.
                cx = (ind % out["heat"].shape[-1]).float() + off_p[:, 0]
                cy = torch.div(ind, out["heat"].shape[-1], rounding_mode="floor").float() + off_p[:, 1]
                w, h = size_p[:, 0].exp(), size_p[:, 1].exp()
                pred_box = torch.stack([cx * stride - w / 2, cy * stride - h / 2,
                                        cx * stride + w / 2, cy * stride + h / 2], dim=1)
                parts["giou"] = utils.box_giou_loss(pred_box, batch["box"][idx]).mean()
                del mask_pos

        w = dict(heat=cfg.w_heat, size=cfg.w_size, off=cfg.w_off, giou=cfg.w_giou,
                 bce=cfg.w_bce, dice=cfg.w_dice, cls=cfg.w_cls)
        if self.uw is None:
            total = sum(w[k] * v for k, v in parts.items())
        else:
            grouped = {
                "det": sum(w[k] * parts[k] for k in ("heat", "size", "off", "giou") if k in parts),
                "seg": sum(w[k] * parts[k] for k in ("bce", "dice") if k in parts),
                "cls": parts.get("cls", zero),
            }
            total = self.uw({k: v for k, v in grouped.items() if torch.is_tensor(v)})

        logs = {f"loss/{k}": float(v.detach()) for k, v in parts.items()}
        logs["loss/total"] = float(total.detach())
        logs["batch/n_annotated"] = n_ann
        return total, logs


def gather_at(dense: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(n,C,Hs,Ws) + flat cell index (n,) -> (n,C) values at that cell."""
    n, c = dense.shape[0], dense.shape[1]
    return dense.reshape(n, c, -1).gather(2, ind.view(n, 1, 1).expand(n, c, 1)).squeeze(-1)


def build_det_targets(boxes: torch.Tensor, hw: tuple[int, int], stride: int, device):
    """Gaussian centre heatmap + size/offset regression targets for one batch of boxes.

    ``boxes`` are absolute pixels in the network input frame. The heatmap radius follows the
    CornerNet rule so that any box whose centre lands inside the Gaussian still overlaps the
    ground truth by >= 0.7 IoU — i.e. the "soft" positives are genuinely near-correct.
    """
    Hs, Ws = int(hw[0]), int(hw[1])
    n = boxes.shape[0]
    heat = torch.zeros((n, 1, Hs, Ws), device=device)
    size_t = torch.zeros((n, 2), device=device)
    off_t = torch.zeros((n, 2), device=device)
    ind = torch.zeros((n,), dtype=torch.long, device=device)
    valid = torch.ones((n,), dtype=torch.bool, device=device)

    b = boxes.detach().float().cpu().numpy()
    import numpy as np
    heat_np = np.zeros((n, Hs, Ws), dtype=np.float32)
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in b[i])
        w, h = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
        cx, cy = (x1 + x2) / 2.0 / stride, (y1 + y2) / 2.0 / stride
        cxi, cyi = int(min(max(cx, 0), Ws - 1)), int(min(max(cy, 0), Hs - 1))
        r = max(0, int(utils.gaussian_radius(h / stride, w / stride)))
        utils.draw_gaussian(heat_np[i], cxi, cyi, r)
        size_t[i, 0], size_t[i, 1] = float(math.log(w)), float(math.log(h))
        off_t[i, 0], off_t[i, 1] = float(cx - cxi), float(cy - cyi)
        ind[i] = cyi * Ws + cxi
    heat = torch.from_numpy(heat_np).unsqueeze(1).to(device)
    return heat, size_t, off_t, ind, valid


# ======================================================================================
# EMA
# ======================================================================================
class ModelEMA:
    """Exponential moving average of weights, evaluated instead of the live model.

    Cheap variance reduction: with a cosine schedule the last few epochs still wobble, and on
    a 20k-frame dataset that wobble is comparable to the ablation effects we are trying to
    measure. The decay is warmed up so the average is not anchored to the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ======================================================================================
# schedule
# ======================================================================================
def lr_at(step: int, steps_per_epoch: int, cfg: Config) -> float:
    warm = cfg.warmup_epochs * steps_per_epoch
    total = cfg.epochs * steps_per_epoch
    if step < warm:
        return cfg.lr * (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    cos = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return cfg.lr * (cfg.min_lr_factor + (1 - cfg.min_lr_factor) * cos)


def param_groups(model: nn.Module, wd: float) -> list[dict]:
    """No weight decay on norms and biases — decaying a BatchNorm gain toward zero is a
    silent capacity cut, and it matters more on a small dataset than a large one."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


# ======================================================================================
# main
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from ('auto' = <out>/last.pt)")
    # overrides (None means "leave whatever the config says")
    for k, t in [("index_path", str), ("test_index_path", str), ("photometric", str), ("norm", str),
                 ("select_on", str), ("select_metric", str), ("tag", str), ("loss_weighting", str)]:
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    for k in ("epochs", "batch_size", "num_workers", "seed", "eval_every", "save_every"):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int, default=None)
    for k in ("lr", "weight_decay", "aug_strength", "width", "ema_decay"):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=float, default=None)
    ap.add_argument("--cpr-stages", dest="cpr_stages", default=None,
                    help="comma-separated subset of CPR stages (ablation block B)")
    ap.add_argument("--no-mask-attn-pool", action="store_true")
    ap.add_argument("--heads", default=None, help="comma-separated subset of det,seg,cls")
    ap.add_argument("--img-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    ap.add_argument("--set", dest="set_kv", action="append", default=[], metavar="KEY=VALUE",
                    help="override ANY Config field, e.g. --set w_giou=0.0 --set warmup_epochs=5. "
                         "Values are parsed as YAML/JSON scalars, so 0.0, true, [384,288] and "
                         "null all work. This exists so every field is reachable from the CLI: "
                         "an ablation that cannot be expressed as a command line is an ablation "
                         "that will be run by editing a config and then mis-recorded.")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="run N training steps and one eval, then exit (smoke test)")
    a = ap.parse_args(argv)

    ov = {k: getattr(a, k) for k in
          ("index_path", "test_index_path", "photometric", "norm", "select_on", "select_metric",
           "tag", "loss_weighting", "epochs", "batch_size", "num_workers", "seed",
           "eval_every", "save_every", "lr", "weight_decay", "aug_strength", "width",
           "ema_decay")}
    if a.cpr_stages is not None:
        ov["cpr_stages"] = [s for s in a.cpr_stages.split(",") if s]
    if a.no_mask_attn_pool:
        ov["use_mask_attn_pool"] = False
    if a.heads is not None:
        ov["heads"] = tuple(s for s in a.heads.split(",") if s)
    if a.img_size is not None:
        ov["img_size"] = tuple(a.img_size)
    for kv in a.set_kv:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            import yaml as _y
            parsed = _y.safe_load(v)
        except Exception:
            parsed = v
        ov[k.strip()] = parsed
    cfg = load_config(a.config, ov)

    out = Path(a.out)
    (out).mkdir(parents=True, exist_ok=True)
    utils.set_seed(cfg.seed, cfg.deterministic)
    device = torch.device(a.device)

    # ---- data -----------------------------------------------------------------------
    loaders = build_loaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    steps_per_epoch = max(1, len(train_loader))

    # ---- model ----------------------------------------------------------------------
    model = HandNet(n_classes=10, norm=cfg.norm, width=cfg.width,
                    use_mask_attn_pool=cfg.use_mask_attn_pool, heads=cfg.heads).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    crit = MultiTaskLoss(cfg).to(device)
    params = param_groups(model, cfg.weight_decay)
    if crit.uw is not None:
        params.append({"params": list(crit.uw.parameters()), "weight_decay": 0.0})
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999))
    ema = ModelEMA(model, cfg.ema_decay)
    use_amp = cfg.amp and device.type == "cuda"

    start_epoch, best, gstep = 0, -1e9, 0
    resume = a.resume
    if resume == "auto":
        resume = str(out / "last.pt") if (out / "last.pt").exists() else None
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"] + 1
        best, gstep = ck.get("best", -1e9), ck.get("gstep", 0)
        print(f"[train] resumed {resume} at epoch {start_epoch}", flush=True)

    with open(out / "config.json", "w") as f:
        json.dump({**asdict(cfg), "n_params": n_par, "device": str(device)}, f, indent=2)
    log_path = out / "log.jsonl"
    print(f"[train] {n_par/1e6:.2f}M params | {len(train_loader.dataset)} train frames "
          f"| {steps_per_epoch} steps/epoch | photometric={cfg.photometric} norm={cfg.norm}",
          flush=True)

    # Imported here, not at module scope: evaluate.py imports nothing from train.py, so this
    # one-way edge keeps the dependency acyclic and lets --dry-run fail fast on data problems
    # before paying for the import.
    from src.evaluate import evaluate_model

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        meters = {}
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            lr = lr_at(gstep, steps_per_epoch, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                outs = model(batch["image"])
                loss, logs = crit(outs, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            gstep += 1
            for k, v in logs.items():
                meters.setdefault(k, utils.AverageMeter()).update(float(v))
            if a.dry_run and i + 1 >= a.dry_run:
                break
            if i % 50 == 0:
                print(f"  e{epoch} {i}/{steps_per_epoch} lr={lr:.2e} "
                      f"loss={logs['loss/total']:.4f}", flush=True)

        rec = {"epoch": epoch, "lr": lr, "time_s": round(time.time() - t0, 1),
               **{f"train/{k.split('/', 1)[1]}": round(m.avg, 5) for k, m in meters.items()}}

        if val_loader is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1
                                       or a.dry_run):
            vm = evaluate_model(ema.ema, val_loader, device, amp=use_amp)
            rec.update({f"val/{k}": v for k, v in _flat_headline(vm).items()})
            if cfg.select_on == "pseudo":
                pm = evaluate_model(ema.ema, val_loader, device, amp=use_amp, pseudo_target=True)
                rec.update({f"pseudo/{k}": v for k, v in _flat_headline(pm).items()})
                sel_src = pm
            else:
                sel_src = vm
            score = _selection_score(sel_src, cfg.select_metric)
            rec["select/score"] = round(score, 5)
            if score > best:
                best = score
                _save(out / "best.pt", model, ema, opt, cfg, epoch, best, gstep, n_par,
                      with_optimizer=False)
                rec["select/is_best"] = True

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[epoch] " + json.dumps(rec), flush=True)

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            _save(out / "last.pt", model, ema, opt, cfg, epoch, best, gstep, n_par)
        if a.dry_run:
            print("[train] dry run complete", flush=True)
            break

    print(f"[train] done. best {cfg.select_metric} on {cfg.select_on} = {best:.5f}", flush=True)
    return 0


def _flat_headline(m: dict) -> dict:
    """The subset of evaluate_model's output worth putting on one log line per epoch."""
    f = m.get("frame", m)
    keep = ("det_acc@0.5", "mean_box_iou", "seg_iou_hand", "seg_miou", "seg_dice",
            "cls_top1", "cls_macro_f1", "cls_ece")
    return {k: round(float(f[k]), 5) for k in keep if k in f}


def _selection_score(m: dict, which: str) -> float:
    f = m.get("frame", m)
    if which == "combined":
        # Equal weight on the three task families so selection cannot be carried by whichever
        # head happens to converge first. All three are in [0,1].
        return (float(f.get("cls_macro_f1", 0.0)) + float(f.get("seg_iou_hand", 0.0))
                + float(f.get("det_acc@0.5", 0.0))) / 3.0
    return float(f.get(which, 0.0))


def _save(path: Path, model, ema, opt, cfg: Config, epoch: int, best: float,
          gstep: int, n_par: int, with_optimizer: bool = True) -> None:
    """Write a checkpoint atomically.

    ``with_optimizer=False`` for ``best.pt``. The optimizer state is 76 MB of a 153 MB
    checkpoint (two moment buffers per parameter) and exists only so training can resume --
    which reads ``last.pt``. Nothing ever resumes from ``best.pt``, and every evaluation reads
    ``ema``. Writing it there doubled the study's disk footprint and, at 39 runs, exhausted the
    workspace quota mid-queue. Learned the hard way.
    """
    tmp = path.with_suffix(".tmp")
    payload = {"model": model.state_dict(), "ema": ema.ema.state_dict(),
               "cfg": asdict(cfg), "epoch": epoch, "best": best, "gstep": gstep,
               "n_params": n_par, "torch": torch.__version__}
    if with_optimizer:
        payload["opt"] = opt.state_dict()
    torch.save(payload, tmp)

    # os.replace is atomic with respect to a CRASH, and that is all it is atomic with respect
    # to. On a filesystem that has hit its quota, torch.save can return without raising and
    # leave a short file, and os.replace will then install that short file over a good
    # checkpoint. That is not hypothetical: it destroyed one of this study's checkpoints.
    # A torch checkpoint is a zip archive, and a truncated one fails on exactly one thing --
    # locating the central directory at the end of the file. Reading the name list is O(1),
    # needs no tensor loading, and catches precisely that failure, so it is cheap enough to do
    # on every write.
    try:
        with zipfile.ZipFile(tmp) as z:
            if not z.namelist():
                raise RuntimeError("empty archive")
    except Exception as e:
        os.remove(tmp)
        raise RuntimeError(
            f"refusing to install {path.name}: the temp file did not survive the write "
            f"({type(e).__name__}: {e}). The existing {path.name} is untouched. "
            f"This almost always means the filesystem is full.") from e
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
