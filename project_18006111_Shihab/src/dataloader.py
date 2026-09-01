"""Dataset and DataLoader construction for the packed RealSense / smartphone releases.

Reads the ``index.json`` written by ``tools/pack_dataset.py`` - one packed JPEG per frame at
512x384, an optional packed PNG mask, and a box already expressed in packed-image pixels.  The
record fields and the ``ann: null`` convention are the packer's; this module is the only reader
and deliberately mirrors it field for field.

What a sample is
----------------
``{"image", "mask", "box", "label", "has_mask"}``.  ``has_mask``
is the contract that keeps unannotated frames honest: the brief's training release annotates
only a couple of keyframes per clip while every frame carries a clip-level gesture label,
so an unannotated frame must still reach the classifier while being
invisible to the detection and segmentation losses.  Those frames get a **zero mask and a zero
box** purely so that ``default_collate`` has something to stack - they are not targets, and
anything that consumes ``box``/``mask`` without first filtering on ``has_mask`` is a bug.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import utils
from src.augment import AugmentPolicy
from src.utils import IMAGENET_MEAN, IMAGENET_STD, load_json, mask_to_box

__all__ = ["HandGestureDataset", "EmptySplitError", "build_loaders", "worker_init_fn"]

MASK_THRESH = 128

# Process-wide, at import, and deliberately so.  OpenCV's pthread pool is not fork-safe: once
# the *parent* has run any threaded cv2 call, `fork`ing a DataLoader worker leaves the child
# holding the pool's mutexes and the first cv2 call in that worker deadlocks - silently, with
# no traceback and no timeout.  Verified here: a parent that reads a dozen samples before
# building a `num_workers=2` loader hangs forever without this line and runs in 0.1 s with it.
# `setNumThreads(0)` does *not* fix it (0 means "auto", not "none"); 1 does, whether set before
# or after the parent's cv2 work.  Importing this module is the declaration that the process
# intends to fork data workers, so this is the honest place for it - and per-image threading
# is worthless anyway when the parallelism is already one process per worker.
cv2.setNumThreads(1)


class EmptySplitError(ValueError):
    """The requested split matches no record.  Its own type so ``build_loaders`` can skip a
    split a pack genuinely does not contain without also swallowing a malformed index."""


def _resize(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to ``size=(W,H)``, area-averaging on the way down and bilinear on the way up."""
    h, w = a.shape[:2]
    if (w, h) == size:
        return a
    interp = cv2.INTER_AREA if size[0] * size[1] < w * h else cv2.INTER_LINEAR
    return cv2.resize(a, size, interpolation=interp)


class HandGestureDataset(Dataset):
    """One packed split.

    Args:
        index_path: path to ``index.json``.
        split: value to match in the index's ``split_map`` - ``"train"``/``"val"`` for a
            training pack, ``"test"`` for one packed with ``--test`` (which maps every
            contributor to ``"test"``), ``"phone"`` for a smartphone pack whose ``split_map``
            names that split.  An unmatched split raises rather than yielding an empty dataset,
            because a silently empty val loader looks exactly like a val loss of 0.
        img_size: ``(W, H)`` of the returned tensors.
        aug: an :class:`~src.augment.AugmentPolicy`, or ``None`` for a fully deterministic
            read (validation and test: no randomness anywhere, just a resize).
        return_meta: attach ``{"clip_key", "subject", "frame"}``, which ``evaluate.py`` needs
            to aggregate metrics per clip as well as per frame.
    """

    def __init__(self, index_path: str, split: str, img_size: tuple[int, int] = (384, 288),
                 aug: AugmentPolicy | None = None, return_meta: bool = False) -> None:
        index = load_json(index_path)
        self.index_path = str(index_path)
        # Prefer the index file's own directory over the recorded absolute ``root``: packing
        # happens on one machine and training on another, so the stored path is often stale.
        here = Path(index_path).expanduser().resolve().parent
        self.root = here if (here / "rgb").is_dir() else Path(index.get("root", here))
        split_map: dict[str, str] = index.get("split_map", {})
        self.records = [r for r in index["records"] if split_map.get(r["subject"]) == split]
        if not self.records:
            raise EmptySplitError(
                f"split {split!r} matches no record in {index_path}; "
                f"split_map assigns {sorted(set(split_map.values()))}")
        self.split, self.img_size, self.aug, self.return_meta = split, tuple(img_size), aug, return_meta
        self.packed_size = tuple(index.get("packed_size", (512, 384)))
        # (x/255 - mean)/std folded into a per-channel 256-entry table: one cv2.LUT does the
        # uint8->float32 cast and the normalisation together, exactly and in ~0.1 ms.
        self._norm_lut = (((np.arange(256, dtype=np.float32)[:, None] / 255.0)
                           - np.float32(IMAGENET_MEAN)) / np.float32(IMAGENET_STD)
                          ).reshape(256, 1, 3).astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, rel: str, flags: int) -> np.ndarray:
        path = self.root / rel
        a = cv2.imread(str(path), flags)
        if a is None:
            raise FileNotFoundError(f"unreadable packed file {path}")
        return a

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        img = cv2.cvtColor(self._read(rec["rgb"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        has_mask = rec.get("ann") is not None
        mask: np.ndarray | None = None
        box: np.ndarray | None = None
        if has_mask:
            m = self._read(rec["ann"], cv2.IMREAD_GRAYSCALE)
            mask = np.where(m >= MASK_THRESH, np.uint8(255), np.uint8(0))
            box = (np.asarray(rec["box"], dtype=np.float32) if rec.get("box")
                   else mask_to_box(mask))

        if self.aug is not None:
            img, mask, box = self.aug(img, mask, box)

        W, H = self.img_size
        img = _resize(img, (W, H))

        if has_mask and mask is not None:
            mask = _resize(mask, (W, H)) >= MASK_THRESH        # bool {False,True}
            if mask.any():
                # Recompute from the *resized* mask rather than scaling the box: it keeps the
                # two exactly consistent at the resolution the loss sees, which is the same
                # reason pack_dataset.py derives its box from the resized mask.  ``.view`` is a
                # free reinterpretation of the bool buffer as {0,1} uint8.
                box = mask_to_box(mask.view(np.uint8), thresh=1)
            else:
                # A near-degenerate hand can
                # vanish under crop+downscale.  Demote to classification-only rather than
                # emitting a degenerate target; this is the *only* case where has_mask is
                # False for a frame that does carry an annotation.
                has_mask = False
        if not has_mask or mask is None:
            mask = np.zeros((H, W), dtype=bool)
            box = np.zeros(4, dtype=np.float32)

        out: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(
                cv2.LUT(img, self._norm_lut).transpose(2, 0, 1))),
            "mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "box": torch.from_numpy(np.asarray(box, dtype=np.float32).reshape(4)),
            "label": torch.tensor(int(rec["cls"]), dtype=torch.int64),
            "has_mask": bool(has_mask),
        }
        if self.return_meta:
            out["meta"] = {
                "clip_key": f"{rec['subject']}/{rec['gesture']}/{rec['clip']}",
                "subject": rec["subject"], "frame": int(rec["frame"]),
                # Carried so evaluate.py can write a per-image prediction row that names the
                # exact source frame. Costs three strings per item and only when return_meta
                # is on, which is eval-only - training never builds this dict.
                "id": str(rec.get("id", "")), "rgb": str(rec.get("rgb", "")),
                "gesture": str(rec.get("gesture", "")),
            }
        return out


def worker_init_fn(worker_id: int) -> None:
    """Per-worker setup.  Registered on every loader, including the eval ones.

    Two jobs.  (1) Pin OpenCV to one thread: each worker is already a process, and cv2's
    default pool would spawn ``n_cores`` threads *per worker* and spend more time contending
    than augmenting.  (2) Seed the global ``random``/``numpy``/``torch`` RNGs from
    ``torch.initial_seed()``, which the DataLoader has already set to ``base_seed + worker_id``
    with a fresh ``base_seed`` drawn from the loader's generator each epoch.  Deriving from it
    rather than from a constant is what keeps the per-worker *and* per-epoch entropy that
    ``AugmentPolicy._rng`` reads back out; seeding with ``seed + worker_id`` instead would
    freeze every epoch to the same augmentations.
    """
    cv2.setNumThreads(1)
    utils.set_seed((torch.initial_seed() + worker_id) % (2 ** 31))


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def build_loaders(cfg: Any) -> dict[str, DataLoader]:
    """Build the ``train`` / ``val`` / ``test`` loaders that the given config can support.

    ``cfg`` may be a dict or any object with these attributes; every key is optional except
    ``index_path``:

    ==================  =========================================================
    ``index_path``      index.json for the training pack (train + val splits)
    ``test_index_path`` index.json for a pack made with ``--test`` (or a phone pack)
    ``splits``          which of ``("train","val","test")`` to build; default all available
    ``img_size``        ``(W,H)``, default ``(384,288)``
    ``batch_size``      default 32; ``eval_batch_size`` defaults to it
    ``num_workers``     default 4
    ``pin_memory``      default ``torch.cuda.is_available()``
    ``persistent_workers`` default ``num_workers > 0``
    ``prefetch_factor``  default None (torch's own default)
    ``drop_last``       train only, default True
    ``seed``            default 0 - seeds the train loader's generator *and* the policy
    ``geometric``       default True
    ``photometric``     default ``"jitter"``; ``"cpr"`` for E3
    ``cpr_stages``      iterable of stage names, or None for all (block-B ablations)
    ``aug_strength``    default 1.0
    ``return_meta``     default False
    ==================  =========================================================

    There is no ``"phone"`` key: a smartphone pack made with ``pack_dataset.py --test`` names
    its split ``"test"``, so it is loaded by pointing ``test_index_path`` at it.  A pack whose
    ``split_map`` genuinely says ``"phone"`` is read directly with
    ``HandGestureDataset(index, "phone", ...)``.

    The train sampler is **plain shuffling**.  No class-balanced sampler:
    measures the classes as exactly uniform (23 clips / 345 frames each), so a balancing
    sampler would resample nothing while adding an unexplained confounder to every ablation.
    """
    index_path = _get(cfg, "index_path") or _get(cfg, "index")
    if not index_path:
        raise ValueError("cfg needs an 'index_path' pointing at a packed index.json")
    test_index = _get(cfg, "test_index_path")
    img_size = tuple(_get(cfg, "img_size", (384, 288)))
    nw = int(_get(cfg, "num_workers", 4))
    bs = int(_get(cfg, "batch_size", 32))
    ebs = int(_get(cfg, "eval_batch_size", bs))
    seed = int(_get(cfg, "seed", 0))
    meta = bool(_get(cfg, "return_meta", False))
    want = _get(cfg, "splits") or ("train", "val", "test")

    stages = _get(cfg, "cpr_stages")
    aug = AugmentPolicy(geometric=bool(_get(cfg, "geometric", True)),
                        photometric=str(_get(cfg, "photometric", "jitter")),
                        cpr_stages=None if stages is None else set(stages),
                        strength=float(_get(cfg, "aug_strength", 1.0)),
                        seed=seed)

    common: dict[str, Any] = dict(
        num_workers=nw, worker_init_fn=worker_init_fn,
        pin_memory=bool(_get(cfg, "pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(_get(cfg, "persistent_workers", nw > 0)) and nw > 0,
    )
    pf = _get(cfg, "prefetch_factor")
    if pf is not None and nw > 0:
        common["prefetch_factor"] = int(pf)

    sources = {"train": (index_path, aug), "val": (index_path, None),
               "test": (test_index or index_path, None)}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        if split not in want:
            continue
        path, policy = sources[split]
        if not path or not os.path.exists(path):
            continue
        try:
            ds = HandGestureDataset(path, split, img_size, policy, return_meta=meta)
        except EmptySplitError:
            continue  # this pack simply has no such split (a --test pack has no train/val)
        if split == "train":
            # One generator, seeded once: it drives the shuffle *and* the per-epoch base_seed
            # that every worker's augmentation stream is derived from.
            gen = torch.Generator().manual_seed(seed)
            loaders[split] = DataLoader(ds, batch_size=bs, shuffle=True, generator=gen,
                                        drop_last=bool(_get(cfg, "drop_last", True)), **common)
        else:
            loaders[split] = DataLoader(ds, batch_size=ebs, shuffle=False, drop_last=False,
                                        **common)
    if not loaders:
        raise ValueError(f"no split of {tuple(want)} could be built from {index_path}")
    return loaders
