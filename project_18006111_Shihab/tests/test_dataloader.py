"""Tests for `src/dataloader.py`. Plain asserts, no pytest.

Run from the project root:
    python tests/test_dataloader.py

A small synthetic release is generated and then packed with the *real*
``tools/pack_dataset.py``, so the reader is tested against the writer rather than against an
idea of what the writer produces.  Some frames deliberately have no annotation, which is the
``ann: null`` case the ``has_mask`` flag exists for.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))

from src.augment import AugmentPolicy  # noqa: E402
from src.dataloader import (  # noqa: E402
    EmptySplitError, HandGestureDataset, build_loaders, worker_init_fn,
)
from src.utils import IMAGENET_MEAN, IMAGENET_STD, load_json, mask_to_box  # noqa: E402

GESTURES3 = ["G01_call", "G02_dislike", "G03_like"]
IMG_SIZE = (160, 120)          # small: these tests are about plumbing, not resolution
_PACK: dict[str, str] = {}


# --------------------------------------------------------------------------- fixtures
def _write_raw(root: str) -> None:
    """A miniature copy of the release layout, including a stray .DS_Store."""
    k = 0
    for si, s in enumerate(("s1", "s2", "s3")):
        for gi, g in enumerate(GESTURES3):
            for ci, c in enumerate(("clip01", "clip04")):
                rgb = os.path.join(root, s, g, c, "rgb")
                ann = os.path.join(root, s, g, c, "annotation")
                os.makedirs(rgb, exist_ok=True)
                os.makedirs(ann, exist_ok=True)
                for f in range(1, 4):
                    rng = np.random.default_rng(1000 * si + 100 * gi + 10 * ci + f)
                    im = (rng.random((480, 640, 3)) * 80 + 60).astype(np.uint8)
                    m = np.zeros((480, 640), np.uint8)
                    x0, y0 = 150 + 20 * gi + 10 * f, 120 + 15 * ci + 8 * f
                    m[y0:y0 + 150, x0:x0 + 115] = 255
                    im[y0:y0 + 150, x0:x0 + 115] = (200, 160, 140)
                    Image.fromarray(im).save(os.path.join(rgb, f"frame_{f:03d}.png"))
                    if k % 5 != 2:                       # every 5th-ish frame is unannotated
                        Image.fromarray(m, mode="L").save(os.path.join(ann, f"frame_{f:03d}.png"))
                    k += 1
    with open(os.path.join(root, "s1", ".DS_Store"), "wb") as fh:
        fh.write(b"junk")


def packs() -> dict[str, str]:
    """Build (once) a train/val pack and a --test pack; return their index.json paths."""
    if _PACK:
        return _PACK
    tmp = tempfile.mkdtemp(prefix="hgtest_")
    atexit.register(shutil.rmtree, tmp, True)
    raw = os.path.join(tmp, "raw")
    _write_raw(raw)
    packer = os.path.join(ROOT, "..", "tools", "pack_dataset.py")
    for name, extra in (("trainval", ["--split-holdout", "1", "--seed", "0"]), ("test", ["--test"])):
        out = os.path.join(tmp, name)
        r = subprocess.run([sys.executable, packer, "--src", raw, "--out", out,
                            "--packed-size", "512", "384", "--workers", "1"] + extra,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        _PACK[name] = os.path.join(out, "index.json")
    return _PACK


def records(index_path: str, split: str) -> list[dict]:
    idx = load_json(index_path)
    return [r for r in idx["records"] if idx["split_map"].get(r["subject"]) == split]


# --------------------------------------------------------------------------- the dataset
def test_sample_shapes_dtypes_and_ranges() -> None:
    p = packs()["trainval"]
    ds = HandGestureDataset(p, "train", IMG_SIZE, aug=None, return_meta=True)
    assert len(ds) == len(records(p, "train")) > 0
    W, H = IMG_SIZE
    for i in (0, len(ds) // 2, len(ds) - 1):
        s = ds[i]
        assert set(s) == {"image", "mask", "box", "label", "has_mask", "meta"}
        assert s["image"].shape == (3, H, W) and s["image"].dtype == torch.float32
        assert s["mask"].shape == (1, H, W) and s["mask"].dtype == torch.float32
        assert torch.isin(s["mask"], torch.tensor([0.0, 1.0])).all()
        assert s["box"].shape == (4,) and s["box"].dtype == torch.float32
        assert s["label"].dtype == torch.int64 and 0 <= int(s["label"]) < 10
        assert isinstance(s["has_mask"], bool)
        assert torch.isfinite(s["image"]).all()
        # normalised, not raw: ImageNet stats put a [0,255] image roughly in [-2.2, 2.7]
        assert -2.3 < float(s["image"].min()) and float(s["image"].max()) < 2.8
        assert set(s["meta"]) == {"clip_key", "subject", "frame"}
        assert s["meta"]["clip_key"].count("/") == 2


def test_normalisation_is_exactly_the_contract() -> None:
    """Denormalising must reproduce the resized packed JPEG pixel for pixel."""
    p = packs()["trainval"]
    ds = HandGestureDataset(p, "val", IMG_SIZE, aug=None)
    s = ds[0]
    rec = records(p, "val")[0]
    raw = cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(p), rec["rgb"])), cv2.COLOR_BGR2RGB)
    ref = cv2.resize(raw, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ref = (ref - np.float32(IMAGENET_MEAN)) / np.float32(IMAGENET_STD)
    got = s["image"].numpy().transpose(1, 2, 0)
    assert np.abs(got - ref).max() < 1e-5, np.abs(got - ref).max()


def test_has_mask_is_false_exactly_when_ann_is_null() -> None:
    """The flag that stops a zero box being trained on as if it were a target.

    (The dataset also demotes a mask that a crop+downscale erased entirely, but no synthetic
    hand here is small enough for that, so the biconditional is exact on this data.)
    """
    p = packs()["trainval"]
    for split in ("train", "val"):
        ds = HandGestureDataset(p, split, IMG_SIZE, aug=None)
        recs = records(p, split)
        n_null = 0
        for i in range(len(ds)):
            s, rec = ds[i], recs[i]
            expect = rec["ann"] is not None
            assert s["has_mask"] is expect, (split, i, rec["id"])
            if not expect:
                n_null += 1
                assert float(s["mask"].sum()) == 0.0, "an unannotated frame must carry a zero mask"
                assert float(s["box"].abs().sum()) == 0.0, "...and a zero box"
            else:
                assert float(s["mask"].sum()) > 0.0
                ref = mask_to_box((s["mask"][0].numpy() > 0.5).astype(np.uint8), thresh=1)
                assert np.abs(s["box"].numpy() - ref).max() <= 1e-4, (s["box"], ref)
                b = s["box"].numpy()
                assert 0 <= b[0] < b[2] <= IMG_SIZE[0] and 0 <= b[1] < b[3] <= IMG_SIZE[1]
        assert n_null > 0, "the fixture must contain unannotated frames or this proves nothing"


def test_aug_none_is_bit_identical_across_reads() -> None:
    p = packs()["trainval"]
    a = HandGestureDataset(p, "val", IMG_SIZE, aug=None)
    b = HandGestureDataset(p, "val", IMG_SIZE, aug=None)
    for i in (0, 3, len(a) - 1):
        for k in ("image", "mask", "box", "label"):
            assert torch.equal(a[i][k], b[i][k]), k
            assert torch.equal(a[i][k], a[i][k]), k


def test_aug_changes_the_sample_and_keeps_box_on_mask() -> None:
    p = packs()["trainval"]
    for mode in ("jitter", "cpr"):
        ds = HandGestureDataset(p, "train", IMG_SIZE, aug=AugmentPolicy(photometric=mode, seed=2))
        i = next(j for j in range(len(ds)) if records(p, "train")[j]["ann"] is not None)
        a, b = ds[i], ds[i]
        assert not torch.equal(a["image"], b["image"]), f"{mode}: two reads were identical"
        for s in (a, b):
            assert s["has_mask"] and float(s["mask"].sum()) > 0
            ref = mask_to_box((s["mask"][0].numpy() > 0.5).astype(np.uint8), thresh=1)
            assert np.abs(s["box"].numpy() - ref).max() <= 1e-4


def test_unknown_split_raises() -> None:
    p = packs()["trainval"]
    try:
        HandGestureDataset(p, "test", IMG_SIZE)
        raise AssertionError("a split with no records must raise, not be silently empty")
    except EmptySplitError as e:
        assert "train" in str(e) and "val" in str(e)
    # the --test pack maps every contributor to "test"
    ds = HandGestureDataset(packs()["test"], "test", IMG_SIZE)
    assert len(ds) == 54


# ------------------------------------------------------------------------ worker seeding
def test_module_pins_opencv_to_one_thread() -> None:
    """Not a style preference: OpenCV's pthread pool is not fork-safe, and a parent that has
    used it deadlocks every forked DataLoader worker on its first cv2 call.  Reproduced and
    fixed in src/dataloader.py; this asserts the fix is still in place."""
    assert cv2.getNumThreads() == 1


def test_parent_cv2_work_then_forked_workers() -> None:
    """The exact sequence that deadlocked: read samples in the parent, then fork workers."""
    idx = packs()["trainval"]
    warm = HandGestureDataset(idx, "train", IMG_SIZE, aug=AugmentPolicy(photometric="cpr", seed=1))
    for i in range(8):
        warm[i]
    assert len(_epoch(_loader(idx, 4, workers=2))) == 4



def _epoch(dl: DataLoader) -> list[torch.Tensor]:
    return [b["image"].clone() for b in dl]


def _loader(index: str, seed: int, workers: int, persistent: bool = False) -> DataLoader:
    # Fail loudly instead of deadlocking: if the module-level cv2.setNumThreads(1) is ever
    # removed, every fork-based test below hangs forever rather than reporting anything.
    assert cv2.getNumThreads() == 1, "importing src.dataloader must pin OpenCV to one thread"
    ds = HandGestureDataset(index, "train", IMG_SIZE,
                            aug=AugmentPolicy(photometric="cpr", seed=seed))
    # the same dataset index four times: any difference between the four outputs is entirely
    # attributable to the augmentation stream, not to the sample.
    sub = Subset(ds, [0, 0, 0, 0])
    return DataLoader(sub, batch_size=1, shuffle=False, num_workers=workers,
                      worker_init_fn=worker_init_fn, persistent_workers=persistent and workers > 0,
                      generator=torch.Generator().manual_seed(seed))


def test_workers_do_not_share_one_rng_stream() -> None:
    """The classic fork bug: an rng built in __init__ gives every worker the same stream.

    With ``num_workers=2`` and no shuffling, worker 0 serves items 0 and 2 and worker 1 serves
    items 1 and 3 -- all of them dataset index 0.  A shared stream makes items 0 and 1 the
    *same* augmentation, and only two distinct images come out instead of four.
    """
    idx = packs()["trainval"]
    outs = _epoch(_loader(idx, 5, workers=2))
    assert len(outs) == 4
    assert not torch.equal(outs[0], outs[1]), (
        "worker 0 and worker 1 produced identical augmentations of the same index "
        "-- the generator is being shared across the fork")
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(outs[i], outs[j]), f"samples {i} and {j} are identical"


def test_fixed_seed_reproduces_a_batch_across_runs() -> None:
    idx = packs()["trainval"]
    a = _epoch(_loader(idx, 5, workers=2))
    b = _epoch(_loader(idx, 5, workers=2))
    c = _epoch(_loader(idx, 6, workers=2))
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "same seed did not replay"
    assert not any(torch.equal(x, y) for x, y in zip(a, c)), "different seeds gave the same batch"


def test_epochs_differ_without_persistent_workers() -> None:
    """Non-persistent workers are re-forked each epoch; without the per-epoch base seed they
    would replay epoch 1's augmentations for the whole run."""
    dl = _loader(packs()["trainval"], 9, workers=2, persistent=False)
    e1, e2 = _epoch(dl), _epoch(dl)
    assert not any(torch.equal(x, y) for x, y in zip(e1, e2))


def test_seeding_also_works_with_no_worker_init_fn() -> None:
    """AugmentPolicy must not depend on worker_init_fn having been registered."""
    ds = HandGestureDataset(packs()["trainval"], "train", IMG_SIZE,
                            aug=AugmentPolicy(photometric="jitter", seed=3))
    dl = DataLoader(Subset(ds, [0, 0, 0, 0]), batch_size=1, num_workers=2,
                    generator=torch.Generator().manual_seed(0))
    outs = _epoch(dl)
    assert not torch.equal(outs[0], outs[1])


def test_single_process_stream_still_advances() -> None:
    dl = _loader(packs()["trainval"], 5, workers=0)
    outs = _epoch(dl)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(outs[i], outs[j])
    assert all(torch.equal(x, y) for x, y in zip(outs, _epoch(_loader(packs()["trainval"], 5, 0))))


# ------------------------------------------------------------------------- build_loaders
def test_build_loaders_from_dict_and_object() -> None:
    p, t = packs()["trainval"], packs()["test"]
    cfg = dict(index_path=p, test_index_path=t, img_size=IMG_SIZE, batch_size=4,
               eval_batch_size=8, num_workers=0, seed=1, photometric="cpr", return_meta=True)
    ls = build_loaders(cfg)
    assert set(ls) == {"train", "val", "test"}
    assert ls["train"].batch_size == 4 and ls["val"].batch_size == 8
    assert ls["train"].drop_last is True and ls["val"].drop_last is False
    assert ls["train"].dataset.aug is not None and ls["val"].dataset.aug is None
    assert ls["test"].dataset.aug is None and len(ls["test"].dataset) == 54
    # plain shuffling, no balanced sampler (the classes are uniform by construction)
    assert isinstance(ls["train"].sampler, torch.utils.data.RandomSampler)
    b = next(iter(ls["train"]))
    assert b["image"].shape == (4, 3, IMG_SIZE[1], IMG_SIZE[0])
    assert b["has_mask"].dtype == torch.bool and b["has_mask"].shape == (4,)
    assert b["label"].shape == (4,) and b["box"].shape == (4, 4)
    assert isinstance(b["meta"]["clip_key"], list) and len(b["meta"]["clip_key"]) == 4

    ls2 = build_loaders(SimpleNamespace(index_path=p, img_size=IMG_SIZE, num_workers=0,
                                        batch_size=2, splits=("train", "val")))
    assert set(ls2) == {"train", "val"}
    assert ls2["train"].dataset.aug.photometric == "jitter"   # documented default


def test_build_loaders_shuffles_train_and_not_val() -> None:
    p = packs()["trainval"]
    cfg = dict(index_path=p, img_size=IMG_SIZE, batch_size=6, num_workers=0, seed=0,
               photometric="none", geometric=False, drop_last=False, splits=("train", "val"))
    ls = build_loaders(cfg)
    order = lambda dl: torch.cat([b["label"] for b in dl])          # noqa: E731
    t1, t2 = order(ls["train"]), order(ls["train"])
    assert not torch.equal(t1, t2), "the train loader must reshuffle each epoch"
    assert torch.equal(order(ls["val"]), order(ls["val"])), "the val loader must not shuffle"
    assert sorted(t1.tolist()) == sorted(t2.tolist())


def test_build_loaders_errors_are_useful() -> None:
    try:
        build_loaders({})
        raise AssertionError("a cfg with no index_path must raise")
    except ValueError as e:
        assert "index_path" in str(e)
    try:
        build_loaders(dict(index_path=packs()["test"], splits=("train",), num_workers=0))
        raise AssertionError("a --test pack has no train split")
    except ValueError:
        pass


def report_timing() -> None:
    cv2.setNumThreads(1)
    p = packs()["trainval"]
    # The synthetic frames are near-uniform noise, so their packed JPEGs are ~229 kB and cost
    # ~3 ms to decode; a real 512x384 frame is ~20 kB and decodes in ~0.6 ms.  Report the
    # decode separately so the augmentation cost can be read off without that distortion.
    first = os.path.join(os.path.dirname(p), load_json(p)["records"][0]["rgb"])
    for _ in range(8):
        cv2.imread(first, cv2.IMREAD_COLOR)
    t0 = time.perf_counter()
    for _ in range(20):
        cv2.imread(first, cv2.IMREAD_COLOR)
    dec = 1000 * (time.perf_counter() - t0) / 20
    print("\n--- dataset timing (single cv2 thread, ms/sample, 384x288 out) ---")
    print(f"  of which JPEG decode       : {dec:6.2f}   "
          f"({os.path.getsize(first) // 1024} kB synthetic frame)")
    for mode in ("none", "jitter", "cpr"):
        aug = None if mode == "none" else AugmentPolicy(photometric=mode, seed=0)
        ds = HandGestureDataset(p, "train", (384, 288), aug=aug)
        for _ in range(8):
            ds[0]
        runs = []
        for _ in range(4):
            t0 = time.perf_counter()
            for i in range(24):
                ds[i % len(ds)]
            runs.append(1000 * (time.perf_counter() - t0) / 24)
        label = "aug=None (val/test path)" if mode == "none" else f'photometric="{mode}"'
        print(f"  {label:26s} : {min(runs):6.2f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if not failed:
        report_timing()
    sys.exit(1 if failed else 0)
