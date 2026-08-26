#!/usr/bin/env bash
# Are 25150455_Guan and "25150455_Guan 2" the same submission?
# Same student number under two folder names is a subject-leakage hazard: if one lands in
# train and the other in val, the val score is inflated by a near-duplicate of a training
# subject. Before deciding how to merge them we need to know whether the pixels differ.
exec 2>&1
cd /var/tmp/cw1_$USER || exit 1
rm -rf guanchk && mkdir -p guanchk
7z x -y -oguanchk raw/rgb_only.7z "rgb_only/25150455_Guan/*"   > /dev/null || exit 1
7z x -y -oguanchk raw/rgb_only.7z "rgb_only/25150455_Guan 2/*" > /dev/null || exit 1
A=guanchk/rgb_only/25150455_Guan
B="guanchk/rgb_only/25150455_Guan 2"
for d in "$A" "$B"; do
  echo "--- $d"
  echo "    png files : $(find "$d" -name '*.png' | wc -l)"
  echo "    rgb       : $(find "$d" -path '*/rgb/*' -name '*.png' | wc -l)"
  echo "    annotation: $(find "$d" -path '*/annotation/*' -name '*.png' | wc -l)"
  echo "    clips     : $(find "$d" -type d -name 'clip*' | wc -l)"
  echo -n "    content-hash (paths relative, sorted): "
  (cd "$d" && find . -name '*.png' | sed 's|^\./||' | sort | xargs -d '\n' md5sum 2>/dev/null \
     | awk '{print $1, $2}' | md5sum | awk '{print $1}')
  echo -n "    pixel-only hash (ignores filenames): "
  (cd "$d" && find . -name '*.png' -print0 | xargs -0 md5sum 2>/dev/null \
     | awk '{print $1}' | sort | md5sum | awk '{print $1}')
done
echo "--- per-file diff on G01_call/clip01/rgb ---"
diff <(ls "$A/G01_call/clip01/rgb" 2>/dev/null) <(ls "$B/G01_call/clip01/rgb" 2>/dev/null) && echo "same filenames"
for f in frame_001.png frame_008.png; do
  a=$(md5sum "$A/G01_call/clip01/rgb/$f" 2>/dev/null | awk '{print $1}')
  b=$(md5sum "$B/G01_call/clip01/rgb/$f" 2>/dev/null | awk '{print $1}')
  echo "    $f  A=$a  B=$b  $( [ "$a" = "$b" ] && echo IDENTICAL || echo DIFFERENT )"
done
rm -rf guanchk
echo "DUPECHECK_DONE"
