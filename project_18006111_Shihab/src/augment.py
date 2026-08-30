"""Augmentation policy: geometric block + swappable photometric block.

Scope
-----
Implements `02_DESIGN.md` s6 (geometric), s7 (Camera-Pipeline Randomisation) and s8 block D
(competing randomisers).  Nothing here imports ``torchvision``: every operation is written
against numpy / OpenCV / PIL, which is what the brief requires (LSA p10-p11).

Conventions
-----------
* Images are ``uint8 (H,W,3)`` **RGB** on the way in and out.  OpenCV is BGR-native, so every
  cv2 call that cares about channel order (only the JPEG codec does) converts explicitly.
* Masks are ``uint8 (H,W)`` with values ``{0,255}``, binarised at 128 (`01_DATA.md` s2).
* Boxes are ``float32 (4,)`` ``(x1,y1,x2,y2)`` with **exclusive** ``x2/y2``, absolute pixels in
  the frame of the image they accompany -- the convention fixed by ``src/utils``.

Sizing contract (matters for the dataloader)
--------------------------------------------
The geometric block returns the **raw crop**, at whatever size the random-resized-crop
rectangle happens to be; it does *not* resize back.  The "resized" half of random-resized-crop
is completed by the dataloader's single resize to ``img_size``.  That keeps the whole pipeline
to exactly one resampling step for the crop plus one warp for the rotation, instead of the
crop->resize->resize double resample a size-preserving policy would force.  It is also why
`02_DESIGN.md` s5 packs at 512x384 rather than 384x288: at ``scale>=0.56`` the crop is still a
downsample at the final 384x288.

Worker seeding (read this before touching ``_rng``)
---------------------------------------------------
A ``np.random.default_rng`` built in ``__init__`` and inherited through ``fork`` gives every
DataLoader worker the *identical* stream, so N workers produce N copies of the same
augmentation and the effective augmentation diversity silently drops by a factor of N.
``_rng`` therefore (re)builds the generator whenever
``(os.getpid(), worker_id, torch.initial_seed())`` changes:

* ``os.getpid()``    -- forces a rebuild inside a forked child, so no stream is ever shared.
* ``worker_id``      -- decorrelates workers within one epoch.
* ``torch.initial_seed()`` -- DataLoader sets this to ``base_seed + worker_id`` with a fresh
  ``base_seed`` drawn from the loader's generator **every epoch**, so the stream also changes
  from epoch to epoch even when ``persistent_workers=False`` re-forks identical workers.  Seed
  the loader's ``generator`` and the whole thing is reproducible run to run.

With ``num_workers=0`` the key never changes, the single generator simply keeps advancing, and
epochs differ for that reason instead.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.utils import mask_to_box

try:  # augment.py stays usable without torch (offline visualisation of the policy)
    import torch
    from torch.utils.data import get_worker_info as _get_worker_info
except Exception:  # pragma: no cover - torch is a hard dependency of the project proper
    torch = None

    def _get_worker_info():  # type: ignore[misc]
        return None


__all__ = [
    "AugmentPolicy", "CPR_STAGES", "PHOTOMETRIC_MODES",
    "geometric_transform", "color_jitter", "cpr", "randconv", "aprs", "augmix",
    "pseudo_target_transform",
]

# Stage names for `02_DESIGN.md` s7, in pipeline order.  Frozen by INTERFACES.md.
CPR_STAGES: tuple[str, ...] = (
    "wb", "exposure", "ccm", "noise", "gamma", "tone",
    "satihue", "sharpblur", "resample", "jpeg", "chroma",
)
PHOTOMETRIC_MODES: tuple[str, ...] = ("none", "jitter", "cpr", "randconv", "aprs", "augmix")

# sRGB<->linear is modelled as a plain 2.2 power (s7 stage 1), not the piecewise IEC curve:
# the stage exists to put stages 2-5 in a roughly linear space, and the piecewise toe changes
# nothing that matters at 8 bits.  Input is uint8, so it is exactly a 256-entry LUT.
_SRGB_TO_LINEAR: np.ndarray = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
# uint8 -> float32 [0,1] as a LUT: cv2.LUT is 0.18 ms on a 512x384 frame against 0.28 ms for
# ``astype(np.float32) / 255`` and it is the identity tone curve when a stage is switched off.
_U8_TO_F32: np.ndarray = (np.arange(256, dtype=np.float32) / 255.0)
_U8_GRID: np.ndarray = (np.arange(256, dtype=np.float64) / 255.0)

# Rec.709-ish luma weights used by the SVG `feColorMatrix` saturate/hueRotate definitions.
_LUM_R, _LUM_G, _LUM_B = 0.213, 0.715, 0.072

_RESAMPLE_KERNELS: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
}
_DOWN_KERNELS = tuple(_RESAMPLE_KERNELS)
# cv2 silently degrades INTER_AREA to INTER_NEAREST when upscaling, which would double the
# effective probability of a nearest-neighbour upsample; drop it from the up-kernel set.
_UP_KERNELS = tuple(k for k in _RESAMPLE_KERNELS if k != "area")

_NOISE_BANK_N = 1 << 21  # 2M float32 = 8 MB per worker; see _NoiseBank


# ======================================================================================
# colour primitives (shared deliberately by the E1 jitter baseline and CPR stage 8)
# ======================================================================================
def _saturation_matrix(s: float) -> np.ndarray:
    """SVG ``feColorMatrix type="saturate"`` -- scales chroma about the luma axis."""
    return np.array([
        [_LUM_R + (1 - _LUM_R) * s, _LUM_G * (1 - s), _LUM_B * (1 - s)],
        [_LUM_R * (1 - s), _LUM_G + (1 - _LUM_G) * s, _LUM_B * (1 - s)],
        [_LUM_R * (1 - s), _LUM_G * (1 - s), _LUM_B + (1 - _LUM_B) * s],
    ], dtype=np.float32)


def _hue_matrix(turns: float) -> np.ndarray:
    """SVG ``feColorMatrix type="hueRotate"`` -- rotation of the chroma plane by ``turns``.

    Deliberately *not* an HSV-cone hue shift.  A round trip through ``cv2.cvtColor`` costs
    3.3 ms on a 512x384 float32 frame against 0.20 ms for a 3x3 ``cv2.transform``; at 16x the
    price the HSV version would dominate the whole CPR budget.  It is also the more physical
    choice for a stage that models a vendor colour-rendering matrix.  The E1 jitter baseline
    uses the same primitive, so the E1/E3 comparison is not confounded by the difference.
    """
    a = 2.0 * math.pi * float(turns)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [_LUM_R + c * (1 - _LUM_R) - s * _LUM_R,
         _LUM_G * (1 - c) - s * _LUM_G,
         _LUM_B * (1 - c) + s * (1 - _LUM_B)],
        [_LUM_R * (1 - c) + s * 0.143,
         _LUM_G + c * (1 - _LUM_G) + s * 0.140,
         _LUM_B * (1 - c) - s * 0.283],
        [_LUM_R * (1 - c) - s * (1 - _LUM_R),
         _LUM_G * (1 - c) + s * _LUM_G,
         _LUM_B + c * (1 - _LUM_B) + s * _LUM_B],
    ], dtype=np.float32)


def _compose(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]):
    """``b`` applied after ``a``, both affine ``(M, bias)`` colour maps."""
    return b[0] @ a[0], b[0] @ a[1] + b[1]


def _apply_colour(img: np.ndarray, m: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """One ``cv2.transform`` with a 3x4 matrix: matrix multiply plus per-channel offset."""
    return cv2.transform(img, np.concatenate([m, bias.reshape(3, 1)], axis=1).astype(np.float32))


def _tone_curve(rng: np.random.Generator, strength: float, at: np.ndarray) -> np.ndarray:
    """Monotone tone curve from 5 perturbed control points (s7 stage 7), sampled at ``at``.

    Monotonicity is enforced by rebuilding the curve from **clamped-positive differences**
    before interpolation.  A non-monotone LUT inverts local contrast over part of the range,
    which no camera does and which would teach the network that a dark-to-light edge and a
    light-to-dark edge are the same thing.  Piecewise-linear interpolation between monotone
    knots is itself monotone, so nothing downstream can reintroduce the problem; and because
    ``at`` is itself monotone in the CPR call site, so is the composed table.
    """
    xs = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    ys = np.clip(xs + rng.uniform(-0.10, 0.10, size=5) * strength, 0.0, 1.0)
    d = np.maximum(np.diff(ys), 0.0)          # a flat run is highlight/shadow clipping: physical
    ys = np.clip(np.concatenate([ys[:1], ys[0] + np.cumsum(d)]), 0.0, 1.0)
    return np.interp(at, xs, ys).astype(np.float32)


def _tone_lut(rng: np.random.Generator, strength: float) -> np.ndarray:
    """The stage-7 curve as a plain 256-entry LUT over a uniform [0,1] grid."""
    return _tone_curve(rng, strength, _U8_GRID)


class _NoiseBank:
    """Pool of N(0,1) samples, read back through random windows.

    Drawing a fresh 512x384x3 Gaussian field costs ~6 ms -- more than the entire rest of the
    CPR chain -- so the stage would dominate the dataloader.  A per-worker pool sampled at a
    random offset costs ~1 us.  The noise is i.i.d. *within* a frame, which is what stage 5
    models; across frames two windows overlap only by chance (2M-element pool, ~1.5M distinct
    offsets), and the field is subsequently scaled by a per-pixel signal-dependent sigma and
    pushed through tone, resample and JPEG, so nothing frame-invariant survives.
    """

    def __init__(self, rng: np.random.Generator, n: int = _NOISE_BANK_N) -> None:
        self.buf = rng.standard_normal(n, dtype=np.float32)

    def draw(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        n = int(np.prod(shape))
        if n > self.buf.size:  # unexpectedly large frame: grow once rather than fail
            self.buf = np.resize(self.buf, n * 2)
        off = int(rng.integers(0, self.buf.size - n + 1))
        return self.buf[off:off + n].reshape(shape)


# ======================================================================================
# geometric block (02_DESIGN.md s6)
# ======================================================================================
def _crop_params(h: int, w: int, rng: np.random.Generator, scale: tuple[float, float],
                 ratio: tuple[float, float]) -> tuple[int, int] | None:
    """One random-resized-crop size proposal, or None if it does not fit."""
    target = h * w * float(rng.uniform(*scale))
    ar = math.exp(float(rng.uniform(math.log(ratio[0]), math.log(ratio[1]))))
    cw, ch = int(round(math.sqrt(target * ar))), int(round(math.sqrt(target / ar)))
    return (cw, ch) if 0 < cw <= w and 0 < ch <= h else None


def geometric_transform(
    img: np.ndarray, mask: np.ndarray | None, box: np.ndarray | None,
    rng: np.random.Generator, *,
    scale: tuple[float, float] = (0.5, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    max_rot_deg: float = 10.0,
    flip_p: float = 0.5,
    min_visible: float = 0.5,
    max_attempts: int = 10,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Random-resized-crop + rotation + horizontal flip, applied jointly to img/mask/box.

    Args:
        min_visible: reject a crop that keeps less than this fraction of the *original* mask
            area.  Default 0.5.  The class label is clip-level, so a crop that keeps a sliver
            of palm still carries the label "three" -- half the hand is the point below which
            the gesture stops being identifiable and the sample becomes label noise.  Raising
            it towards 1.0 collapses random-resized-crop to a near-identity centre crop;
            1.3 % of real frames are already border-truncated (`01_DATA.md` s2.5), so partial
            hands are in-distribution and a value near 0 is not obviously wrong either -- it is
            a parameter precisely because the right value is arguable.
        max_attempts: bounded rejection sampling; on exhaustion the identity crop is used, so
            the function always returns something valid.

    The rotated box is recomputed as the tight box of the **rotated mask**, never as the extent
    of the four rotated corners: at 10 degrees the corner-extent box is up to ~9 % larger in
    each dimension than the true box, and a detector trained on inflated targets is
    systematically wrong.  The corner-extent fallback is used only when no mask exists (an
    unannotated frame), where nothing better is available; the resulting box is an
    over-estimate and such samples carry ``has_mask=False`` downstream anyway.
    """
    h, w = img.shape[:2]
    angle = float(rng.uniform(-max_rot_deg, max_rot_deg))
    rotate = abs(angle) > 1e-3

    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0) if rotate else None

    mask_r: np.ndarray | None = None
    need = 0
    if mask is not None:
        if rotate:
            # BORDER_CONSTANT 0 for the mask: reflecting it would paste a phantom second hand
            # against the border and the recomputed box would grow to cover it.
            mr = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_r = np.where(mr >= 128, np.uint8(255), np.uint8(0))
        else:
            mask_r = mask
        need = int(math.ceil(min_visible * int(np.count_nonzero(mask >= 128))))

    cw, ch, x0, y0 = w, h, 0, 0
    for _ in range(max_attempts):
        prop = _crop_params(h, w, rng, scale, ratio)
        if prop is None:
            continue
        pw, ph = prop
        px = int(rng.integers(0, w - pw + 1))
        py = int(rng.integers(0, h - ph + 1))
        if mask_r is not None and need > 0:
            vis = int(np.count_nonzero(mask_r[py:py + ph, px:px + pw]))
            if vis < max(1, need):
                continue
        cw, ch, x0, y0 = pw, ph, px, py
        break

    if rotate:
        # Fold "rotate about the centre" and "translate the crop origin to 0" into one warp so
        # the image is resampled exactly once, directly at the crop size.  Slicing the
        # full-frame warped mask by the same rectangle is geometrically identical (verified to
        # within the interpolator's fixed-point quantum).
        aff = (np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
               @ np.vstack([rot, [0.0, 0.0, 1.0]]))[:2]
        # BORDER_REPLICATE for the image: black wedges would be a perfect "this sample was
        # augmented" cue, and reflection would invent a second hand.
        out_img = cv2.warpAffine(img, aff, (cw, ch), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        aff = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0]])
        out_img = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    out_mask = None if mask_r is None else np.ascontiguousarray(mask_r[y0:y0 + ch, x0:x0 + cw])

    flip = bool(rng.random() < flip_p)
    if flip:
        # Slice-flip both, so image and mask cannot disagree about which axis was mirrored.
        out_img = np.ascontiguousarray(out_img[:, ::-1])
        if out_mask is not None:
            out_mask = np.ascontiguousarray(out_mask[:, ::-1])

    out_box: np.ndarray | None = None
    if out_mask is not None and out_mask.any():
        out_box = mask_to_box(out_mask)
    elif out_mask is None and box is not None:
        pts = np.array([[box[0], box[1]], [box[2], box[1]],
                        [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
        pts = pts @ aff[:, :2].T + aff[:, 2]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        if flip:
            x1, x2 = cw - x2, cw - x1
        x1, x2 = float(np.clip(x1, 0, cw - 1)), float(np.clip(x2, 1, cw))
        y1, y2 = float(np.clip(y1, 0, ch - 1)), float(np.clip(y2, 1, ch))
        out_box = np.array([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)], dtype=np.float32)

    if out_mask is not None and out_box is None:
        # Every accepted crop keeps >= min_visible of the mask, so this is unreachable for a
        # non-empty input mask; if the input mask was already empty, hand back the input.
        return img, mask, box
    return out_img, out_mask, out_box


# ======================================================================================
# photometric block
# ======================================================================================
def color_jitter(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
                 brightness: float = 0.4, contrast: float = 0.4,
                 saturation: float = 0.4, hue: float = 0.1) -> np.ndarray:
    """E1 baseline: conventional brightness / contrast / saturation / hue jitter.

    All four are affine in RGB, so they are composed into a single 3x4 matrix and applied with
    one ``cv2.transform``.  The four factors are still drawn and *composed in a random order*,
    matching the reference ColorJitter semantics; the one approximation is that contrast pivots
    on the luma mean of the incoming image rather than being recomputed after each preceding
    op, which moves the pivot by a fraction of a level.
    """
    x = cv2.LUT(img, _U8_TO_F32)
    grey = float(_LUM_R * x[:, :, 0].mean() + _LUM_G * x[:, :, 1].mean() + _LUM_B * x[:, :, 2].mean())
    eye, zero = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    b = 1.0 + strength * float(rng.uniform(-brightness, brightness))
    c = 1.0 + strength * float(rng.uniform(-contrast, contrast))
    ops = [
        (eye * b, zero),
        (eye * c, np.full(3, grey * (1.0 - c), dtype=np.float32)),
        (_saturation_matrix(1.0 + strength * float(rng.uniform(-saturation, saturation))), zero),
        (_hue_matrix(strength * float(rng.uniform(-hue, hue))), zero),
    ]
    acc = (eye, zero)
    for i in rng.permutation(len(ops)):
        acc = _compose(acc, ops[int(i)])
    out = _apply_colour(x, acc[0], acc[1])
    return cv2.convertScaleAbs(np.clip(out, 0.0, 1.0, out=out), alpha=255.0)


def cpr(img: np.ndarray, rng: np.random.Generator, stages: Iterable[str] | None = None,
        strength: float = 1.0) -> np.ndarray:
    """Camera-Pipeline Randomisation, `02_DESIGN.md` s7.  Photometric only.

    ``stages=None`` runs all of :data:`CPR_STAGES`; any subset runs only those (this is what
    drives the block-B leave-one-out ablation).  Stage 1 (sRGB->linear) and the *return* to
    display space are not toggleable and always run as an inverse pair: with ``"gamma"``
    disabled the return uses exactly gamma 2.2, so the round trip is the identity and every
    later stage still sees a display-referred image.  That is what makes each named stage
    genuinely removable without breaking its successors.

    Clamping.  Values are clamped to [0,1] exactly once in linear light -- on the way out of
    it, before the display encode -- because that is where a real sensor clips (the ADC) and
    because the CCM and the noise can drive a channel negative, which a fractional power turns
    into NaN.  Between stages 2 and 5 nothing is clamped: highlights above 1.0 there are
    physically meaningful headroom, and clipping them early would flatten every bright region
    before the tone curve ever sees it.  In display space each stage that can overshoot --
    saturation, unsharp, bicubic/Lanczos ringing -- clamps itself, so the uint8 conversion at
    the end never has to saturate.
    """
    on = set(CPR_STAGES) if stages is None else set(stages)
    s = float(strength)
    if s <= 0.0:
        # A short-circuit, not a scaled-to-nothing chain.  Two stages have no identity element
        # inside their specification: the JPEG quality range tops out at 95, which is lossy
        # (~60 levels of ringing on a hard edge), and a chromatic-aberration shift is +/-1 px
        # or nothing -- there is no smaller shift.  Scaling their parameters toward zero
        # therefore lands on "mildest", not "identity", so strength 0 is defined as the block
        # being switched off.  The corollary is worth knowing for the s7.1 strength sweep:
        # strength interpolates the *continuous* stages only; the two discrete ones sit at
        # their mildest setting for every strength above 0 and are otherwise removed through
        # ``cpr_stages``.
        return img.copy()

    # -- stage 1: sRGB -> linear (fused with the uint8 -> float32 conversion) ------------
    x = cv2.LUT(img, _SRGB_TO_LINEAR)

    # -- stages 2-4: white balance, exposure, CCM.  All three are linear maps, so they are
    #    folded into one 3x3 and applied with a single cv2.transform.  Folding costs nothing
    #    in fidelity and keeps each one independently skippable.
    m = np.eye(3, dtype=np.float32)
    if "ccm" in on:
        ccm = np.eye(3, dtype=np.float32) + rng.normal(0.0, 0.05 * s, size=(3, 3)).astype(np.float32)
        rs = ccm.sum(axis=1, keepdims=True)          # rows sum to 1 => neutral grey stays neutral
        ccm /= np.where(np.abs(rs) < 1e-3, np.float32(1.0), rs)
        m = ccm @ m
    if "wb" in on:                                   # WB precedes the CCM in a real ISP, and
        g = np.ones(3, dtype=np.float32)             # m * g[None,:] == ccm @ diag(g) does that
        g[0] = 1.0 + s * (float(rng.uniform(0.80, 1.25)) - 1.0)
        g[2] = 1.0 + s * (float(rng.uniform(0.80, 1.25)) - 1.0)
        m = m * g[None, :]
    if "exposure" in on:
        m = m * np.float32(2.0 ** (s * float(rng.uniform(-0.6, 0.6))))
    if not np.array_equal(m, np.eye(3, dtype=np.float32)):
        x = cv2.transform(x, m)

    # -- stage 5: signal-dependent (shot + read) noise ----------------------------------
    if "noise" in on and rng.random() < 0.5:
        a = float(np.exp(rng.uniform(math.log(1e-5), math.log(1e-3)))) * s
        b = float(np.exp(rng.uniform(math.log(1e-6), math.log(1e-4)))) * s
        sig = np.maximum(x, 0.0)                     # CCM can go negative; sqrt of it cannot
        sig *= a
        sig += b
        np.sqrt(sig, out=sig)
        sig *= _bank(rng).draw(x.shape, rng)
        x += sig

    np.clip(x, 0.0, 1.0, out=x)   # the ADC clips; also, a fractional power of a negative is NaN

    # -- stages 6 + 7: linear -> display gamma, then the monotone tone curve.  Both are
    #    per-pixel scalar maps, so they are evaluated once into a single 256-entry table
    #    instead of twice over the image.  The image is quantised into a **gamma-2.0 code**
    #    (one cheap SIMD sqrt) rather than a linear one: 8 bits of *linear* light bands the
    #    shadows badly -- linear step 1/255 is display level 37 -- whereas a gamma-2.0 code has
    #    essentially sRGB's shadow precision, which is all the uint8 input ever carried.  The
    #    residual exponent 2/gamma then rides in the table.  This replaces a 1.25 ms
    #    transcendental pass over the frame with ~5 us of table building.
    gamma = 2.2 if "gamma" not in on else 2.2 + s * (float(rng.uniform(1.8, 2.6)) - 2.2)
    np.sqrt(x, out=x)
    code = _U8_GRID ** (2.0 / gamma)                        # stage 6, on the 256 code values
    lut = _tone_curve(rng, s, code) if "tone" in on else code.astype(np.float32)
    x = cv2.LUT(cv2.convertScaleAbs(x, alpha=255.0), lut)

    # -- stage 8: vendor colour rendering (saturation + hue), one composed matrix -------
    # Saturation is drawn from U(0.6, 1.3), NOT the U(0.6, 1.5) of the design table.
    # The held-out pseudo-target shift (s7.1) applies saturation x1.4; leaving 1.5 as the upper
    # bound would put the pseudo-target inside CPR's reachable set and make target-free model
    # selection circular. Narrowed deliberately -- see docs/02_DESIGN.md s7 note.
    if "satihue" in on:
        mm, bb = _compose(
            (_saturation_matrix(1.0 + s * (float(rng.uniform(0.6, 1.3)) - 1.0)), np.zeros(3, np.float32)),
            (_hue_matrix(s * float(rng.uniform(-0.05, 0.05))), np.zeros(3, np.float32)))
        x = np.clip(_apply_colour(x, mm, bb), 0.0, 1.0)

    # -- stage 9: sharpening vs soft optics.  Two-sided by construction (s7 "Bidirectionality"):
    #    exactly one of the two fires, so the policy covers "phone is sharper than RealSense"
    #    as well as the usual degradation direction.
    if "sharpblur" in on:
        if rng.random() < 0.5:
            sg = 0.3 + s * float(rng.uniform(0.1, 1.2))
            x = cv2.GaussianBlur(x, _ksize(sg), sg)
        else:
            sg = 0.5 + s * float(rng.uniform(0.0, 1.0))
            amt = s * float(rng.uniform(0.3, 1.5))
            x = np.clip(cv2.addWeighted(x, 1.0 + amt, cv2.GaussianBlur(x, _ksize(sg), sg), -amt, 0.0),
                        0.0, 1.0)

    # -- stage 10: resolution / resampling mismatch, kernel randomised on both legs -----
    if "resample" in on and rng.random() < 0.5:
        h, w = x.shape[:2]
        f = 1.0 + s * (float(rng.uniform(1.0, 2.0)) - 1.0)
        dw, dh = max(8, int(round(w / f))), max(8, int(round(h / f)))
        down = _RESAMPLE_KERNELS[_DOWN_KERNELS[int(rng.integers(len(_DOWN_KERNELS)))]]
        up = _RESAMPLE_KERNELS[_UP_KERNELS[int(rng.integers(len(_UP_KERNELS)))]]
        # Independent draws for the two legs: a real chain downscales in the ISP and upscales
        # for display/inference with different filters, and it covers a strictly larger space.
        x = np.clip(cv2.resize(cv2.resize(x, (dw, dh), interpolation=down), (w, h),
                               interpolation=up), 0.0, 1.0)

    # Invariant from here on: x is float32 in [0,1].  Every stage above either preserves the
    # range (LUT, Gaussian blur -- a convex combination) or clips explicitly, so the uint8
    # conversion below never has to saturate and cv2.convertScaleAbs' abs() is never reached.
    out = cv2.convertScaleAbs(x, alpha=255.0)

    # -- stage 11: JPEG re-encode.  cv2 is BGR-native and JPEG luma is 0.299R+0.587G+0.114B,
    #    so feeding RGB in as BGR would put the chroma subsampling on the wrong channel.
    if "jpeg" in on and rng.random() < 0.5:
        q = int(np.clip(round(95 - s * (95 - float(rng.uniform(40, 95)))), 1, 100))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # -- stage 12: lateral chromatic aberration, approximated as opposed +/-1 px R/B shifts.
    #    Real lateral CA is radial (per-channel magnification); a uniform shift is the standard
    #    cheap stand-in and is indistinguishable from it over a 384 px frame at +/-1 px.
    if "chroma" in on and rng.random() < 0.2:
        dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
        if dx or dy:
            out[:, :, 0] = _shift(out[:, :, 0], dx, dy)
            out[:, :, 2] = _shift(out[:, :, 2], -dx, -dy)
    return out


def _ksize(sigma: float) -> tuple[int, int]:
    """Odd kernel size covering ~+/-2.5 sigma.  cv2's ksize=0 uses +/-3 sigma and costs up to
    3x more for a difference that is invisible after the following quantisation."""
    k = 2 * int(2.5 * sigma) + 1
    return (max(3, min(k, 9)),) * 2


def _shift(ch: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a single channel by (dx,dy) with edge replication (never wrap-around)."""
    out = ch
    if dy:
        out = np.pad(out, ((dy, 0) if dy > 0 else (0, -dy), (0, 0)), mode="edge")
        out = out[:ch.shape[0]] if dy > 0 else out[-ch.shape[0]:]
    if dx:
        out = np.pad(out, ((0, 0), (dx, 0) if dx > 0 else (0, -dx)), mode="edge")
        out = out[:, :ch.shape[1]] if dx > 0 else out[:, -ch.shape[1]:]
    return out


_BANKS: dict[int, _NoiseBank] = {}


def _bank(rng: np.random.Generator) -> _NoiseBank:
    """Per-process noise pool, keyed on pid so a forked child never reuses the parent's.

    The pool is filled from a *separate* generator seeded by one draw from ``rng``: filling it
    from ``rng`` itself would pull 2M variates out of the augmentation stream the first time
    the noise stage fires, which would make the stream position depend on which sample happened
    to trigger it.  One draw keeps the pool reproducible without perturbing anything else.
    """
    key = os.getpid()
    if key not in _BANKS:
        _BANKS.clear()  # a forked child inherits the parent's dict; drop it rather than grow
        _BANKS[key] = _NoiseBank(np.random.default_rng(int(rng.integers(1 << 62))))
    return _BANKS[key]


# ======================================================================================
# competing randomisers (02_DESIGN.md s8 block D)
# ======================================================================================
def randconv(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """RandConv -- Xu et al., "Robust and Generalizable Visual Representation Learning via
    Random Convolutions", ICLR 2021.  Random-weight k x k conv, k ~ U{1,3}, He-scaled, output
    mixed with the input by alpha ~ U(0,1)."""
    x = cv2.LUT(img, _U8_TO_F32)
    k = int(rng.choice([1, 3]))
    w = rng.normal(0.0, math.sqrt(2.0 / (k * k * 3)), size=(3, 3, k, k)).astype(np.float32)
    if k == 1:
        y = cv2.transform(x, w.reshape(3, 3))
    else:
        y = np.zeros_like(x)
        for o in range(3):
            for i in range(3):
                y[:, :, o] += cv2.filter2D(x[:, :, i], -1, w[o, i])
    # Match the conv output's first two moments to the input's, as the reference implementation
    # does; without it the He-scaled response is off-scale and the mix is dominated by clipping.
    y = (y - y.mean()) / (y.std() + 1e-6) * (x.std() + 1e-6) + x.mean()
    alpha = 1.0 - strength * float(rng.uniform(0.0, 1.0))
    return cv2.convertScaleAbs(np.clip(alpha * x + (1.0 - alpha) * y, 0.0, 1.0), alpha=255.0)


def aprs(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """APR-S -- Chen et al., "Amplitude-Phase Recombination", ICCV 2021.  Keep the phase
    spectrum of the image, take the amplitude spectrum of a randomly augmented copy of it."""
    x = cv2.LUT(img, _U8_TO_F32)
    y = cv2.LUT(_pil_pool(img, rng, n=2), _U8_TO_F32)
    fx = np.fft.rfft2(x, axes=(0, 1))
    fy = np.fft.rfft2(y, axes=(0, 1))
    amp = np.abs(fy) * strength + np.abs(fx) * (1.0 - strength)
    rec = np.fft.irfft2(amp * fx / (np.abs(fx) + 1e-8), s=x.shape[:2], axes=(0, 1))
    return cv2.convertScaleAbs(np.clip(rec.astype(np.float32), 0.0, 1.0), alpha=255.0)


# AugMix's *published* operation set: it deliberately omits contrast, colour, brightness,
# sharpness, noise and blur so that ImageNet-C stays a clean held-out benchmark.  Those are
# exactly the nuisance variables that separate a RealSense frame from a phone frame, which is
# why D4 is included -- as a method that structurally cannot cover our shift.  Adding the
# missing ops would destroy the point of running it.
def _autocontrast(im: Image.Image, _l: float, _r) -> Image.Image: return ImageOps.autocontrast(im)
def _equalize(im: Image.Image, _l: float, _r) -> Image.Image: return ImageOps.equalize(im)
def _posterize(im, lv, r): return ImageOps.posterize(im, 4 - int(_lvl(lv, 4, r)))
def _solarize(im, lv, r): return ImageOps.solarize(im, 256 - int(_lvl(lv, 256, r)))


def _lvl(level: float, maxval: float, rng: np.random.Generator) -> float:
    """AugMix's ``float_parameter(sample_level(level), maxval)``; callers ``int()`` it where
    the reference implementation uses ``int_parameter``."""
    return float(rng.uniform(0.1, level)) * maxval / 10.0


def _rot(im, lv, r):
    d = _lvl(lv, 30, r)
    return im.rotate(d if r.random() < 0.5 else -d, resample=Image.BILINEAR)


def _shear(im, lv, r, axis: int):
    v = _lvl(lv, 3, r) / 10.0
    v = v if r.random() < 0.5 else -v
    c = (1, v, 0, 0, 1, 0) if axis == 0 else (1, 0, 0, v, 1, 0)
    return im.transform(im.size, Image.AFFINE, c, resample=Image.BILINEAR)


def _trans(im, lv, r, axis: int):
    v = _lvl(lv, im.size[axis] / 3.0, r)
    v = v if r.random() < 0.5 else -v
    c = (1, 0, v, 0, 1, 0) if axis == 0 else (1, 0, 0, 0, 1, v)
    return im.transform(im.size, Image.AFFINE, c, resample=Image.BILINEAR)


# named rather than lambdas so an AugmentPolicy stays picklable under a spawn start method
def _shear_x(im, lv, r): return _shear(im, lv, r, 0)
def _shear_y(im, lv, r): return _shear(im, lv, r, 1)
def _trans_x(im, lv, r): return _trans(im, lv, r, 0)
def _trans_y(im, lv, r): return _trans(im, lv, r, 1)


_AUGMIX_OPS: tuple[Callable, ...] = (
    _autocontrast, _equalize, _posterize, _rot, _solarize,
    _shear_x, _shear_y, _trans_x, _trans_y,
)

#: The four label-safe members of the published set. Used when a mask or box must stay in sync
#: with the image -- see ``augmix``. Note that these four still exclude contrast, colour,
#: brightness, sharpness, noise and blur, so the property D4 is cited for survives the
#: restriction: AugMix structurally cannot cover a camera-pipeline shift either way.
_AUGMIX_PHOTOMETRIC_OPS: tuple[Callable, ...] = (
    _autocontrast, _equalize, _posterize, _solarize,
)


def _pil_pool(img: np.ndarray, rng: np.random.Generator, n: int = 1,
              geometric: bool = True) -> np.ndarray:
    """``n`` random ops from the AugMix set -- also APR-S's "augmented copy" generator.

    ``geometric=False`` restricts the draw to the four photometric ops. See ``augmix``.
    """
    pool = _AUGMIX_OPS if geometric else _AUGMIX_PHOTOMETRIC_OPS
    im = Image.fromarray(img)
    for _ in range(n):
        im = pool[int(rng.integers(len(pool)))](im, 3, rng)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def augmix(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
           width: int = 3, depth: int = -1, severity: int = 3,
           geometric: bool = True) -> np.ndarray:
    """AugMix -- Hendrycks et al., ICLR 2020.  Dirichlet-weighted mix of ``width`` op chains.

    **Documented deviation, applied whenever a mask or box accompanies the image.**  Five of the
    nine published ops are geometric (rotate, shear x/y, translate x/y).  AugMix mixes several
    independently-augmented branches with Dirichlet weights, so there is no single geometry to
    apply to the label -- the mixed image is a superposition of differently-warped copies.
    Applying them image-only leaves the mask and box describing content that has moved (measured
    displacement up to 25 px), which is not a caveat but label noise: it would corrupt the
    detection and segmentation targets of the D4 ablation and make its numbers uninterpretable
    for reasons unrelated to AugMix.

    So when the caller has a label to protect (``geometric=False``, set automatically by
    ``AugmentPolicy`` whenever a mask or box is present) the draw is restricted to the four
    photometric ops.  The classification-only case keeps the published set.  This is stated in
    `02_DESIGN.md` s8 block D so the D4 result is read with the deviation in view: the remaining
    four ops still exclude contrast, colour, brightness, sharpness, noise and blur, which is the
    property D4 is cited for.
    """
    x = cv2.LUT(img, _U8_TO_F32)
    ws = rng.dirichlet([1.0] * width).astype(np.float32)
    m = float(rng.beta(1.0, 1.0)) * float(strength)
    mix = np.zeros_like(x)
    for i in range(width):
        d = depth if depth > 0 else int(rng.integers(1, 4))
        mix += ws[i] * cv2.LUT(_pil_pool(img, rng, n=d, geometric=geometric), _U8_TO_F32)
    return cv2.convertScaleAbs(np.clip((1.0 - m) * x + m * mix, 0.0, 1.0), alpha=255.0)


# ======================================================================================
# held-out pseudo-target transform (02_DESIGN.md s7.1)
# ======================================================================================
# DELIBERATE DUPLICATION.  Everything below is written standalone -- its own literal control
# points, its own saturation matrix, its own unsharp arithmetic -- and calls no CPR helper.
# If it reused `_tone_lut` or `_saturation_matrix`, the "held-out transform family" claim in
# s7.1 would be false: CPR could generate the exact operating point that all CPR strength and
# variant selection is scored against, and the target-free protocol would be circular.
# The S-curve is deliberately *stronger than CPR can draw* -- as a whole curve, not point by
# point.  Over 200 CPR draws no draw tracks the composed pseudo-target transfer curve to within
# 0.04 at every input level (closest 0.067), and that whole-curve separation is what
# tests/test_augment.py asserts.  At either individual control point the pseudo-target value is
# still inside CPR's realisable range: CPR spans [0.098, 0.447] at input 0.25 where this table
# sits at 0.110, and [0.503, 0.979] at 0.75 where it sits at 0.890.  Without the whole-curve
# margin the "held-out family" is only held out on paper -- a 2000-draw search over CPR's
# stage 7 found curves within 0.015 of an earlier, milder version of this table.
_PT_TONE_X = np.array([0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00])
_PT_TONE_Y = np.array([0.00, 0.030, 0.110, 0.500, 0.890, 0.970, 1.00])
_PSEUDO_TONE_LUT: np.ndarray = np.interp(
    np.linspace(0.0, 1.0, 256), _PT_TONE_X, _PT_TONE_Y).astype(np.float32)
_PSEUDO_SAT_M: np.ndarray = np.array([  # saturate(1.4), written out rather than derived
    [0.213 + 0.787 * 1.4, 0.715 * (1 - 1.4), 0.072 * (1 - 1.4)],
    [0.213 * (1 - 1.4), 0.715 + 0.285 * 1.4, 0.072 * (1 - 1.4)],
    [0.213 * (1 - 1.4), 0.715 * (1 - 1.4), 0.072 + 0.928 * 1.4],
], dtype=np.float32)


def pseudo_target_transform(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """The held-out shift used for target-free model selection (`02_DESIGN.md` s7.1).

    Fixed phone-style tone curve -> 2x bicubic up-resample (and back, because the caller feeds
    the result straight to the model and the size must not change) -> strong unsharp mask ->
    saturation x1.4.  ``rng`` is optional: with ``None`` the stated operating point is used
    exactly; with a generator the unsharp amount and saturation get a +/-10 % jitter so a
    selection sweep sees a small neighbourhood rather than one point.  The *family* is fixed
    either way.
    """
    x = cv2.LUT(img, _PSEUDO_TONE_LUT)
    h, w = x.shape[:2]
    x = cv2.resize(cv2.resize(x, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC),
                   (w, h), interpolation=cv2.INTER_CUBIC)
    amt, sat = 1.5, 1.4
    if rng is not None:
        amt *= 1.0 + float(rng.uniform(-0.1, 0.1))
        sat *= 1.0 + float(rng.uniform(-0.1, 0.1))
    blur = cv2.GaussianBlur(x, (7, 7), 1.2)
    x = cv2.addWeighted(x, 1.0 + amt, blur, -amt, 0.0)
    sm = _PSEUDO_SAT_M if sat == 1.4 else np.array([
        [0.213 + 0.787 * sat, 0.715 * (1 - sat), 0.072 * (1 - sat)],
        [0.213 * (1 - sat), 0.715 + 0.285 * sat, 0.072 * (1 - sat)],
        [0.213 * (1 - sat), 0.715 * (1 - sat), 0.072 + 0.928 * sat]], dtype=np.float32)
    return cv2.convertScaleAbs(np.clip(cv2.transform(x, sm), 0.0, 1.0), alpha=255.0)


# ======================================================================================
# policy
# ======================================================================================
class AugmentPolicy:
    """Geometric block + one photometric block, per `INTERFACES.md`.

    Args:
        geometric: run the s6 crop/rotate/flip block.
        photometric: one of :data:`PHOTOMETRIC_MODES`.
        cpr_stages: subset of :data:`CPR_STAGES` (``None`` = all).  Only meaningful for
            ``photometric="cpr"``; an unknown name raises rather than silently running full
            CPR, because a typo in a block-B leave-one-out row would otherwise produce a
            perfectly plausible but wrong ablation number.
        strength: scales every sampled *photometric* magnitude about its identity value
            (probabilities are left alone); ``0.0`` switches the photometric block off exactly
            -- see :func:`cpr` for why that has to be a switch rather than a limit.  The geometric block
            is deliberately **not** scaled by it: ``strength`` is the knob the s7.1 pseudo-target
            protocol sweeps to pick a CPR operating point, and letting it also widen or narrow
            the crop and rotation ranges would confound that sweep with a geometry change.
            Geometry is tuned, if at all, through ``geometric_transform``'s own arguments.
        seed: base entropy.  See the module docstring for how it combines with the worker id
            and the per-epoch DataLoader seed.
    """

    def __init__(self, geometric: bool = True, photometric: str = "jitter",
                 cpr_stages: set[str] | None = None, strength: float = 1.0,
                 seed: int | None = None) -> None:
        if photometric not in PHOTOMETRIC_MODES:
            raise ValueError(f"photometric must be one of {PHOTOMETRIC_MODES}, got {photometric!r}")
        if cpr_stages is not None:
            bad = set(cpr_stages) - set(CPR_STAGES)
            if bad:
                raise ValueError(f"unknown CPR stage(s) {sorted(bad)}; valid: {CPR_STAGES}")
            cpr_stages = set(cpr_stages)
        self.geometric = bool(geometric)
        self.photometric = photometric
        self.cpr_stages = cpr_stages
        self.strength = float(strength)
        self.seed = 0 if seed is None else int(seed)
        self._key: tuple | None = None
        self._gen: np.random.Generator | None = None

    def _rng(self) -> np.random.Generator:
        """Per-(process, worker, epoch) generator -- see the module docstring."""
        info = _get_worker_info()
        wid = -1 if info is None else int(info.id)
        tseed = int(torch.initial_seed()) if torch is not None else 0
        key = (os.getpid(), wid, tseed)
        if key != self._key or self._gen is None:
            self._key = key
            # The pid is a *cache key only* and is deliberately absent from the entropy: it
            # changes between runs, and putting it in the seed would make every run
            # irreproducible -- the opposite of the bug the pid is here to catch.
            self._gen = np.random.default_rng(np.random.SeedSequence(
                [self.seed & 0xFFFFFFFF, wid & 0xFFFFFFFF, tseed & 0xFFFFFFFFFFFFFFFF]))
        return self._gen

    def __call__(self, img: np.ndarray, mask: np.ndarray | None, box: np.ndarray | None
                 ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        rng = self._rng()
        if self.geometric:
            img, mask, box = geometric_transform(img, mask, box, rng)
        p, s = self.photometric, self.strength
        if p == "jitter":
            img = color_jitter(img, rng, s)
        elif p == "cpr":
            img = cpr(img, rng, self.cpr_stages, s)
        elif p == "randconv":
            img = randconv(img, rng, s)
        elif p == "aprs":
            img = aprs(img, rng, s)
        elif p == "augmix":
            # Geometry-preserving whenever there is a label to keep in sync (see augmix()).
            img = augmix(img, rng, s, geometric=(mask is None and box is None))
        return img, mask, box

    def __repr__(self) -> str:  # shows up in run logs; the ablation row must be identifiable
        st = "all" if self.cpr_stages is None else ",".join(sorted(self.cpr_stages))
        return (f"AugmentPolicy(geometric={self.geometric}, photometric={self.photometric!r}, "
                f"cpr_stages={st}, strength={self.strength}, seed={self.seed})")
