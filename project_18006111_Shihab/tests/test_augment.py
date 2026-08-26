"""Tests for `src/augment.py`. Plain asserts, no pytest.

Run from the project root:
    python tests/test_augment.py

The load-bearing test is `test_geometric_box_matches_mask`: a mask/box desynchronisation is
silent -- training converges, nothing crashes, and every detection number is quietly wrong --
so it is checked over hundreds of random draws rather than once.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import augment as A  # noqa: E402
from src.augment import AugmentPolicy, CPR_STAGES, pseudo_target_transform  # noqa: E402
from src.utils import mask_to_box  # noqa: E402

W, H = 512, 384
RECT = (170, 110, 330, 270)          # x1, y1, x2, y2 of the synthetic "hand", clear of borders


def make_sample(textured: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image whose bright region coincides *exactly* with the mask rectangle."""
    x1, y1, x2, y2 = RECT
    if textured:
        rng = np.random.default_rng(0)
        img = (rng.random((H, W, 3)) * 60 + 20).astype(np.uint8)
    else:
        img = np.zeros((H, W, 3), np.uint8)
    img[y1:y2, x1:x2] = 255
    mask = np.zeros((H, W), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return img, mask, np.array(RECT, dtype=np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(max(union, 1))


# ------------------------------------------------------------------- geometric block
def test_geometric_box_matches_mask() -> None:
    """For every draw, the returned box must be the tight box of the returned mask."""
    img, mask, box = make_sample()
    rng = np.random.default_rng(0)
    n_rot_nontrivial = 0
    for _ in range(400):
        oi, om, ob = A.geometric_transform(img, mask, box, rng)
        assert om is not None and ob is not None
        ref = mask_to_box(om)
        assert ref is not None, "an accepted crop must never empty the mask"
        assert np.abs(ob - ref).max() <= 1.0, (ob, ref)
        # exclusive convention, and the box must sit inside the returned frame
        h, w = om.shape
        assert 0 <= ob[0] < ob[2] <= w and 0 <= ob[1] < ob[3] <= h, (ob, om.shape)
        assert oi.shape[:2] == om.shape and oi.dtype == np.uint8 and om.dtype == np.uint8
        if (om.shape != mask.shape) or not np.array_equal(om, mask):
            n_rot_nontrivial += 1
    assert n_rot_nontrivial > 350, "the geometric block is doing nothing on most draws"


def test_geometric_image_and_mask_stay_aligned() -> None:
    """The *image* must move with the mask, not just the box.

    ``make_sample`` paints the hand region white, so thresholding the transformed image
    recovers where the hand went; if the flip or the crop were applied to only one of the two
    this IoU collapses.  A naive implementation that flips the image but not the mask scores
    ~0.1 here.
    """
    img, mask, box = make_sample(textured=False)
    rng = np.random.default_rng(1)
    worst = 1.0
    for _ in range(200):
        oi, om, _ = A.geometric_transform(img, mask, box, rng)
        bright = oi[:, :, 0] >= 128
        worst = min(worst, iou(bright, om >= 128))
    assert worst > 0.95, f"image/mask alignment IoU dropped to {worst:.3f}"


def test_geometric_preserves_the_hand() -> None:
    """Flip / rotate / crop keep the foreground: it never vanishes and the area stays sane."""
    img, mask, box = make_sample()
    area0 = int((mask >= 128).sum())
    rng = np.random.default_rng(2)
    fracs = []
    for _ in range(300):
        _, om, _ = A.geometric_transform(img, mask, box, rng, min_visible=0.5)
        a = int((om >= 128).sum())
        assert a > 0
        fracs.append(a / area0)
    fracs = np.array(fracs)
    # min_visible=0.5 is a hard floor (a little slack for the rotation's bilinear edge);
    # the crop never magnifies, so nothing can exceed 1.
    assert fracs.min() >= 0.49, fracs.min()
    assert fracs.max() <= 1.01, fracs.max()
    assert fracs.mean() > 0.8, fracs.mean()      # most draws keep nearly all of the hand

    # the parameter has to actually bite
    rng = np.random.default_rng(3)
    strict = np.array([(A.geometric_transform(img, mask, box, rng, min_visible=0.95)[1] >= 128).sum()
                       / area0 for _ in range(200)])
    assert strict.min() >= 0.94, strict.min()


def test_geometric_components() -> None:
    """Each component fires at roughly its stated rate and is individually exact."""
    img, mask, box = make_sample(textured=False)
    rng = np.random.default_rng(4)
    # flip only: no rotation, no crop -> an exact mirror
    flips = 0
    for _ in range(200):
        oi, om, ob = A.geometric_transform(img, mask, box, rng, scale=(1.0, 1.0),
                                           ratio=(4 / 3, 4 / 3), max_rot_deg=0.0)
        assert oi.shape[:2] == (H, W)
        if not np.array_equal(om, mask):
            flips += 1
            assert np.array_equal(om, mask[:, ::-1]), "flip must be an exact mirror"
            assert np.allclose(ob, [W - RECT[2], RECT[1], W - RECT[0], RECT[3]])
        else:
            assert np.allclose(ob, box)
    assert 70 < flips < 130, f"flip rate {flips}/200 is not ~0.5"

    # rotation only: the box of a rotated rectangle must be *smaller* than the extent of its
    # rotated corners -- the specific error the mask-recompute exists to avoid.
    rng = np.random.default_rng(5)
    _, om, ob = A.geometric_transform(img, mask, box, rng, scale=(1.0, 1.0), ratio=(4 / 3, 4 / 3),
                                      max_rot_deg=10.0, flip_p=0.0)
    corners = np.array([[RECT[0], RECT[1]], [RECT[2], RECT[1]],
                        [RECT[2], RECT[3]], [RECT[0], RECT[3]]], float)
    ang = recover_angle(om, mask)
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    p = corners @ rot[:, :2].T + rot[:, 2]
    naive = np.concatenate([p.min(0), p.max(0)])[[0, 1, 2, 3]]
    assert (ob[2] - ob[0]) <= (naive[2] - naive[0]) + 1.0
    assert (ob[3] - ob[1]) <= (naive[3] - naive[1]) + 1.0


def recover_angle(rot_mask: np.ndarray, ref: np.ndarray) -> float:
    """Recover the applied rotation from the mask, for the corner-extent comparison above."""
    best, ba = -1.0, 0.0
    for a in np.arange(-10.0, 10.01, 0.25):
        m = cv2.warpAffine(ref, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0), (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        s = iou(m >= 128, rot_mask >= 128)
        if s > best:
            best, ba = s, a
    return ba


def test_geometric_without_mask_falls_back_to_corners() -> None:
    """No mask -> the box is carried by the affine, over-estimating but never None."""
    img, _, box = make_sample()
    rng = np.random.default_rng(6)
    for _ in range(100):
        oi, om, ob = A.geometric_transform(img, None, box, rng)
        assert om is None and ob is not None and ob.dtype == np.float32
        h, w = oi.shape[:2]
        assert 0 <= ob[0] < ob[2] <= w and 0 <= ob[1] < ob[3] <= h


# --------------------------------------------------------------- photometric blocks
def test_photometric_modes_run_and_leave_geometry_alone() -> None:
    img, mask, box = make_sample()
    for mode in A.PHOTOMETRIC_MODES:
        p = AugmentPolicy(geometric=False, photometric=mode, seed=11)
        for _ in range(4):
            oi, om, ob = p(img, mask, box)
            assert oi.dtype == np.uint8 and oi.shape == (H, W, 3), (mode, oi.shape, oi.dtype)
            assert 0 <= int(oi.min()) and int(oi.max()) <= 255
            # bit-identical: a photometric op that touched the mask or box would be a bug
            assert np.array_equal(om, mask), mode
            assert np.array_equal(ob, box), mode
        if mode != "none":
            # Median over draws, not a single draw. AugMix legitimately returns something very
            # close to the input when the sampled chain lands on near-identity ops and the
            # Dirichlet/Beta mixing weights come out small -- that is the method, not a bug.
            # A single-draw threshold here is flaky; the failure mode we actually want to catch
            # is a photometric mode that is silently a no-op on EVERY draw.
            ds = sorted(float(np.abs(p(img, mask, box)[0].astype(np.int16) - img.astype(np.int16)).mean())
                        for _ in range(9))
            d = ds[len(ds) // 2]
            assert d > 0.5, f"{mode} median change over 9 draws only {d:.3f} levels ({ds})"


def test_photometric_strength_zero_is_identity() -> None:
    img, mask, box = make_sample()
    # several draws each: the probabilistic stages (noise, blur/sharpen, resample, JPEG,
    # chromatic aberration) do not all fire on any one sample, so a single draw can pass by
    # luck -- this test used to, before strength=0 was defined as a hard switch-off.
    for mode in ("jitter", "cpr", "randconv", "aprs", "augmix"):
        p = AugmentPolicy(geometric=False, photometric=mode, strength=0.0, seed=3)
        for _ in range(25):
            d = np.abs(p(img, mask, box)[0].astype(np.int16) - img.astype(np.int16)).max()
            assert d <= 2, f"{mode} at strength=0 moved a pixel by {d} levels"


def test_policy_validates_its_arguments() -> None:
    for bad in ("Jitter", "colour", ""):
        try:
            AugmentPolicy(photometric=bad)
            raise AssertionError(f"photometric={bad!r} should have raised")
        except ValueError:
            pass
    try:
        AugmentPolicy(photometric="cpr", cpr_stages={"wb", "tonemap"})
        raise AssertionError("a typo'd CPR stage must raise, not silently run full CPR")
    except ValueError as e:
        assert "tonemap" in str(e)


# ------------------------------------------------------------------------------ CPR
def test_cpr_all_stages_and_leave_one_out() -> None:
    img, _, _ = make_sample()
    rng = np.random.default_rng(21)
    full = np.array([A.cpr(img, rng).astype(np.int16) for _ in range(8)])
    d = np.abs(full - img.astype(np.int16)).mean()
    assert d > 4.0, f"full CPR only moves the image by {d:.2f} levels -- it is nearly a no-op"
    assert full.std(axis=0).mean() > 2.0, "CPR draws are barely different from each other"

    for drop in CPR_STAGES:                       # block-B leave-one-out must all run
        sub = set(CPR_STAGES) - {drop}
        for _ in range(6):
            o = A.cpr(img, rng, sub)
            assert o.dtype == np.uint8 and o.shape == img.shape and np.isfinite(o).all()
    for only in CPR_STAGES:                       # ...and so must every stage on its own
        for _ in range(6):
            o = A.cpr(img, rng, {only})
            assert o.dtype == np.uint8 and o.shape == img.shape


def test_cpr_empty_stage_set_is_a_clean_round_trip() -> None:
    """sRGB->linear and the return to display must invert each other exactly.

    This is what makes every named stage skippable: with all of them off the chain still runs
    (linearise, quantise into the gamma code, table back out) and must give the input back.
    """
    img, _, _ = make_sample()
    out = A.cpr(img, np.random.default_rng(0), set())
    err = np.abs(out.astype(np.int16) - img.astype(np.int16))
    assert err.max() <= 2 and err.mean() < 0.2, (err.max(), err.mean())


def test_cpr_tone_lut_is_monotone() -> None:
    rng = np.random.default_rng(99)
    for i in range(1000):
        lut = A._tone_lut(rng, 1.0)
        assert lut.shape == (256,) and lut.dtype == np.float32
        assert lut.min() >= 0.0 and lut.max() <= 1.0
        assert np.all(np.diff(lut) >= 0.0), f"draw {i} is not monotone"
    # the composed gamma-then-tone table used inside cpr() must be monotone too
    for gamma in (1.8, 2.2, 2.6):
        code = A._U8_GRID ** (2.0 / gamma)
        for _ in range(200):
            assert np.all(np.diff(A._tone_curve(rng, 1.0, code)) >= 0.0)
    # and the curve must be non-trivial: at least some draws bend away from the identity
    devs = [np.abs(A._tone_lut(rng, 1.0) - A._U8_GRID).max() for _ in range(50)]
    assert max(devs) > 0.03, "the tone curve never moves"


def test_cpr_jpeg_stage_round_trips_in_rgb() -> None:
    """The stage must survive the codec and must not swap R and B (cv2 is BGR-native)."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = 210, 120, 40      # R >> G >> B
    img[100:200, 100:200] = (40, 120, 210)                       # a block with the order reversed
    rng = np.random.default_rng(5)
    fired = 0
    for _ in range(40):
        o = A.cpr(img, rng, {"jpeg"})
        assert o.dtype == np.uint8 and o.shape == img.shape
        means = o.reshape(-1, 3).mean(0)
        ref = img.reshape(-1, 3).mean(0)
        assert np.abs(means - ref).max() < 6.0, (means, ref)      # a channel swap moves these ~90
        if np.abs(o.astype(np.int16) - img.astype(np.int16)).mean() > 0.5:
            fired += 1
    assert 8 < fired < 34, f"JPEG stage fired {fired}/40 times, expected about half"


def test_cpr_stage_effects_are_real() -> None:
    """Every stage, run alone, must move the image on at least some draws."""
    img, _, _ = make_sample()
    base = A.cpr(img, np.random.default_rng(0), set()).astype(np.int16)
    rng = np.random.default_rng(7)
    for st in CPR_STAGES:
        moved = max(float(np.abs(A.cpr(img, rng, {st}).astype(np.int16) - base).mean())
                    for _ in range(30))
        assert moved > 0.2, f"stage {st!r} never changes the image (moved {moved:.4f})"


def test_cpr_noise_bank_is_not_a_constant_field() -> None:
    """Two noisy draws must differ; a bank read at a fixed offset would make them identical."""
    img = np.full((H, W, 3), 128, np.uint8)
    rng = np.random.default_rng(4)
    outs = [A.cpr(img, rng, {"noise"}).astype(np.int16) for _ in range(12)]
    diffs = [np.abs(outs[i] - outs[j]).mean() for i in range(12) for j in range(i + 1, 12)]
    assert max(diffs) > 0.2, "no pair of noise draws differs"


def test_shift_replicates_and_never_wraps() -> None:
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(A._shift(a, 1, 0), np.array([[0, 0, 1, 2], [4, 4, 5, 6], [8, 8, 9, 10]]))
    assert np.array_equal(A._shift(a, -1, 0), np.array([[1, 2, 3, 3], [5, 6, 7, 7], [9, 10, 11, 11]]))
    assert np.array_equal(A._shift(a, 0, 1), np.array([[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7]]))


# ------------------------------------------------------------- held-out pseudo-target
def test_pseudo_target_transform() -> None:
    img, _, _ = make_sample()
    out = pseudo_target_transform(img, None)
    assert out.dtype == np.uint8 and out.shape == img.shape
    d = float(np.abs(out.astype(np.int16) - img.astype(np.int16)).mean())
    assert d > 3.0, f"the pseudo-target barely changes the image ({d:.2f} levels)"
    assert np.array_equal(out, pseudo_target_transform(img, None)), "rng=None must be deterministic"
    # a generator gives a small neighbourhood, not a different family
    r = np.random.default_rng(0)
    jit = pseudo_target_transform(img, r)
    assert np.abs(jit.astype(np.int16) - out.astype(np.int16)).mean() < 12.0

    # ---- held-out claim, measured on the COMPOSED transform ------------------------------
    # The earlier version of this test compared `_tone_lut` against `_PSEUDO_TONE_LUT`, which
    # is the wrong object: `cpr()` applies gamma AFTER the tone LUT, so the curve the network
    # actually sees is the composition, and the composition reaches places the bare LUT does
    # not. Comparing the bare tables made the held-out claim look far safer than it is. This
    # version pushes a ramp through the real `cpr()` and the real `pseudo_target_transform()`
    # and compares the transfer curves they realise end to end.
    assert A._PSEUDO_TONE_LUT is not None and np.all(np.diff(A._PSEUDO_TONE_LUT) >= 0)
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (8, 1))[..., None].repeat(3, 2)

    def curve_of(u8):
        return u8[:, :, 0].mean(0).astype(np.float32) / 255.0

    pt = curve_of(pseudo_target_transform(ramp, None))
    draws = np.stack([curve_of(AugmentPolicy(geometric=False, photometric="cpr", seed=s)
                               (ramp, None, None)[0]) for s in range(200)])
    # (a) no CPR draw reproduces the pseudo-target curve over its whole range
    closest = float(np.abs(draws - pt[None, :]).max(axis=1).min())
    assert closest > 0.04, ("some CPR draw matched the pseudo-target transfer curve to within "
                            f"{closest:.4f} everywhere -- the held-out claim would be false")
    # (b) NOT asserted, and the reason is worth recording: at either individual control point
    # the pseudo-target value IS inside CPR's realisable range (measured over 200 draws, CPR
    # spans [0.098, 0.447] at input 0.25 where the pseudo-target sits at 0.110, and
    # [0.503, 0.979] at input 0.75 where it sits at 0.890). Point-wise separation would need a
    # physically absurd curve -- crushed blacks below anything CPR can reach, or near-clipped
    # highlights above it. What is genuinely held out is therefore the curve *as a whole*,
    # asserted in (a), together with the four-operation composition. `02_DESIGN.md` s7.1 states
    # this in exactly these terms rather than the stronger claim it originally made.


# ---------------------------------------------------------------------- seeding basics
def test_policy_streams_advance_and_are_reproducible() -> None:
    img, mask, box = make_sample()
    a = AugmentPolicy(photometric="cpr", seed=17)
    b = AugmentPolicy(photometric="cpr", seed=17)
    c = AugmentPolicy(photometric="cpr", seed=18)
    xa = [a(img, mask, box)[0] for _ in range(4)]
    xb = [b(img, mask, box)[0] for _ in range(4)]
    xc = [c(img, mask, box)[0] for _ in range(4)]
    assert all(np.array_equal(p, q) for p, q in zip(xa, xb)), "same seed must replay exactly"
    assert not any(np.array_equal(p, q) for p, q in zip(xa, xc)), "different seeds must differ"
    assert not np.array_equal(xa[0], xa[1]), "the stream must advance between samples"


def report_timing() -> None:
    rng = np.random.default_rng(0)
    frame = cv2.resize((rng.random((48, 64, 3)) * 255).astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(frame, (384, 288))
    mask = np.zeros((H, W), np.uint8)
    mask[RECT[1]:RECT[3], RECT[0]:RECT[2]] = 255
    box = np.array(RECT, np.float32)
    cv2.setNumThreads(1)                     # what a DataLoader worker will actually see

    def ms(fn, n=120, warm=30):
        """Best of 4 sub-runs: this box has 2 cores, so the median is contention-inflated."""
        for _ in range(warm):
            fn()
        k, runs = max(1, n // 4), []
        for _ in range(4):
            t0 = time.perf_counter()
            for _ in range(k):
                fn()
            runs.append(1000 * (time.perf_counter() - t0) / k)
        return min(runs)

    print("\n--- timing (single cv2 thread, ms/sample) ---")
    r = np.random.default_rng(1)
    print(f"  cpr()                 512x384 : {ms(lambda: A.cpr(frame, r)):6.2f}")
    print(f"  cpr()                 384x288 : {ms(lambda: A.cpr(small, r)):6.2f}")
    print(f"  color_jitter()        512x384 : {ms(lambda: A.color_jitter(frame, r)):6.2f}")
    print(f"  color_jitter()        384x288 : {ms(lambda: A.color_jitter(small, r)):6.2f}")
    print(f"  pseudo_target()       512x384 : {ms(lambda: A.pseudo_target_transform(frame, r), 60):6.2f}")
    print(f"  randconv()            512x384 : {ms(lambda: A.randconv(frame, r), 60):6.2f}")
    print(f"  augmix()              512x384 : {ms(lambda: A.augmix(frame, r), 20, 5):6.2f}")
    print(f"  aprs()                512x384 : {ms(lambda: A.aprs(frame, r), 12, 4):6.2f}")
    for mode in ("none", "jitter", "cpr"):
        p = AugmentPolicy(photometric=mode, seed=1)
        print(f"  policy(geo+{mode:<7s})        : {ms(lambda p=p: p(frame, mask, box), 80):6.2f}")


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
