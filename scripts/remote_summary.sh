#!/usr/bin/env bash
# Compact status: which runs have results, and their headline numbers.
exec 2>&1
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
"$WS"/../comp0248_lsa >/dev/null 2>&1
/var/tmp/cw1_$USER/venv/cw1/bin/python - <<'PY'
import json, glob, os
WS="/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa"
K=["det_acc@0.5","mean_box_iou","seg_iou_hand","seg_dice","cls_top1","cls_macro_f1","cls_ece"]
files=sorted(glob.glob(f"{WS}/results/*.json"))
print(f"{'run':34s} {'n':>6s} " + " ".join(f"{k.split('_')[-1][:7]:>7s}" for k in K))
for f in files:
    try: d=json.load(open(f))
    except Exception: continue
    fr=d.get("frame",{})
    name=os.path.basename(f)[:-5]
    print(f"{name:34s} {d.get('n_frames',0):6d} " + " ".join(f"{fr.get(k,float('nan')):7.4f}" for k in K))
print()
runs=sorted(os.listdir(f"{WS}/runs")) if os.path.isdir(f"{WS}/runs") else []
for r in runs:
    lp=f"{WS}/runs/{r}/log.jsonl"
    if os.path.exists(lp):
        lines=[l for l in open(lp) if l.strip()]
        if lines:
            last=json.loads(lines[-1])
            print(f"  {r:22s} epoch {last['epoch']:>2}  val_f1={last.get('val/cls_macro_f1','-')}  "
                  f"val_iou={last.get('val/seg_iou_hand','-')}")
PY
