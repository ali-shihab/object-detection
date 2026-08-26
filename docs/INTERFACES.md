# Code interface contract (Cycle 2)

Frozen signatures so modules written in parallel compose. Anything not listed is free.
Python 3.11, torch 2.6, numpy 2.x, opencv-python-headless, PIL. No torchvision.models,
no torchvision.ops, no detection/segmentation libraries.

Package root is `project_18006111_Shihab/`; modules live in `src/` and import each other as
`from src import utils` (the package is run from the project root, `src/__init__.py` exists).

## Canonical types

- Box: `np.ndarray` / `torch.Tensor` shape `(...,4)`, order `(x1, y1, x2, y2)`, **absolute pixels
  in the coordinate frame of the tensor it accompanies**, `x2>x1`, `y2>y1`.
- Mask: `uint8` `(H,W)` on disk with values {0,255}; `float32` `(1,H,W)` in a batch, values {0,1}.
- Image in a batch: `float32` `(3,H,W)`, ImageNet-free normalisation — `(x/255 - mean)/std` with
  `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)` (constants in `utils.IMAGENET_MEAN/STD`).
- Class index: `int` in `[0,10)`, ordered `G01_call .. G10_three` (`utils.GESTURES` is that list).

## `src/utils.py`

```python
GESTURES: list[str]                      # 10 names, index == class id
IMAGENET_MEAN: tuple[float,float,float]
IMAGENET_STD:  tuple[float,float,float]

def set_seed(seed: int, deterministic: bool = False) -> None
def mask_to_box(mask: np.ndarray, thresh: int = 128) -> np.ndarray | None
    # (H,W) uint8 -> (4,) float32 (x1,y1,x2,y2); None if no foreground
def box_iou(a: Tensor, b: Tensor) -> Tensor          # (N,4),(N,4) -> (N,)  elementwise pairs
def box_giou_loss(pred: Tensor, tgt: Tensor) -> Tensor   # (N,4),(N,4) -> (N,)  1 - GIoU
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float
def draw_gaussian(heat: np.ndarray, cx: int, cy: int, radius: int) -> None   # in-place max-merge

# metric primitives, all operating on already-decoded predictions
def seg_scores(pred: Tensor, tgt: Tensor) -> dict   # (N,1,H,W) prob in [0,1], (N,1,H,W) in {0,1}
    # -> {"iou_hand","iou_bg","miou","dice"}  each (N,) per-image tensors
class MetricAccumulator:                             # streams per-frame records, aggregates at end
    def add(self, *, clip_key: str, box_iou: float | None, seg: dict,
            gt_class: int, pred_class: int, cls_conf: float, det_conf: float) -> None
    def summary(self) -> dict                        # per-frame and per-clip blocks, confusion matrix
class AverageMeter: ...
def save_json(obj, path) -> None
def load_json(path)
```

## `src/model.py`

```python
class HandNet(nn.Module):
    def __init__(self, n_classes: int = 10, norm: str = "bn",       # "bn" | "gn" | "ibn"
                 width: float = 1.0, use_mask_attn_pool: bool = True,
                 heads: tuple[str,...] = ("det","seg","cls")) -> None
    def forward(self, x: Tensor) -> dict[str, Tensor]
        # x (B,3,H,W). Returns, with H,W the INPUT size and Hs=H//4, Ws=W//4:
        #   "heat"  (B,1,Hs,Ws)  raw logits
        #   "size"  (B,2,Hs,Ws)  log(w), log(h) in input pixels
        #   "off"   (B,2,Hs,Ws)
        #   "seg"   (B,1,H,W)    raw logits
        #   "cls"   (B,10)       raw logits
        # A head absent from `heads` is simply not in the dict (single-task ablation A-MT).

@torch.no_grad()
def decode_detection(out: dict, k: int = 1) -> tuple[Tensor, Tensor]
    # -> boxes (B,4) absolute input-frame pixels, scores (B,)
```

`width` scales all channel counts. Everything must run under `torch.autocast(bf16)`.

## `src/dataloader.py`

```python
class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, index_path: str, split: str,              # "train" | "val" | "test" | "phone"
                 img_size: tuple[int,int] = (384, 288),          # (W,H)
                 aug: "AugmentPolicy | None" = None,
                 return_meta: bool = False) -> None
    def __getitem__(self, i) -> dict
        # {"image": (3,H,W) float32,
        #  "mask":  (1,H,W) float32 in {0,1},
        #  "box":   (4,)    float32, absolute pixels in the (W,H) frame,
        #  "label": int64 scalar,
        #  "has_mask": bool  -> False means mask/box are undefined and must be excluded from
        #                       det/seg losses and metrics,
        #  "meta": {"clip_key": str, "subject": str, "frame": int}  if return_meta}

def build_loaders(cfg) -> dict[str, DataLoader]
```

`index.json` schema produced by `tools/pack_dataset.py`:

```json
{"root": "<dir containing rgb/ and ann/>",
 "records": [{"id":"<subject>/<gesture>/<clip>/<frame>", "subject":"18006111_Shihab",
              "gesture":"G01_call", "clip":"clip01", "frame":1,
              "rgb":"rgb/....jpg", "ann":"ann/....png" | null,
              "cls":0, "box":[x1,y1,x2,y2] | null, "src_size":[640,480]}],
 "packed_size": [512, 384], "split_map": {"<subject>": "train"|"val"}}
```

Box coordinates in `index.json` are in **packed-image pixels** (512x384), not source pixels.

## `src/augment.py`

```python
class AugmentPolicy:
    def __init__(self, geometric: bool = True, photometric: str = "jitter",
                 # "none" | "jitter" | "cpr" | "randconv" | "aprs" | "augmix"
                 cpr_stages: set[str] | None = None,   # subset of CPR stage names for Block B
                 strength: float = 1.0, seed: int | None = None)
    def __call__(self, img: np.ndarray, mask: np.ndarray | None,
                 box: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        # img uint8 HxWx3 RGB. Geometric ops transform mask and box together; photometric ops
        # touch img only.

CPR_STAGES: tuple[str,...]   # ("wb","exposure","ccm","noise","gamma","tone","satihue",
                             #  "sharpblur","resample","jpeg","chroma")
def pseudo_target_transform(img: np.ndarray, rng) -> np.ndarray
    # the HELD-OUT transform family used for target-free model selection: fixed phone-style tone
    # curve, 2x bicubic up-resample, strong unsharp mask, saturation x1.4. Must share NO code
    # path with the CPR stages so it stays a genuine held-out shift.
```

## `tools/pack_dataset.py`

CLI: `python tools/pack_dataset.py --src <dir-or-7z> --out <dir> --packed-size 512 384
[--split-holdout N] [--seed 0] [--test]`.
Writes `<out>/rgb/*.jpg`, `<out>/ann/*.png`, `<out>/index.json`. Must process one contributor at a
time and delete its extracted originals before moving on (host disk is tight).
