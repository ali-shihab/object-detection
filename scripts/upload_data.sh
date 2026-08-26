#!/bin/zsh
# Ships the two raw datasets to the Knuckles GPU host's node-local scratch.
# Streams the RealSense test tree straight through tar|ssh so no temp copy is
# written to the Mac (13 GB free, 98% full).
set -u
HOST="${1:-skate-l}"
PASS=~/.ssh/.ucl_knuckles_pass
PROXY="ProxyCommand=sshpass -P \"assword:\" -f $PASS ssh -o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -W %h:%p ucl-knuckles"
OPTS=(-o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o "$PROXY")
REMOTE="ashihab@${HOST}.cs.ucl.ac.uk"
RAW=/var/tmp/cw1_ashihab/raw
CW1=~/workspace/projects/ucl/object_detection_and_classification/cw1

echo "[$(date +%T)] mkdir remote"
sshpass -P "assword:" -f $PASS ssh "${OPTS[@]}" "$REMOTE" "mkdir -p $RAW" || exit 1

echo "[$(date +%T)] uploading rgb_only.7z (6.37 GB)"
sshpass -P "assword:" -f $PASS scp "${OPTS[@]}" ~/Downloads/rgb_only.7z "$REMOTE:$RAW/rgb_only.7z" || exit 1
echo "[$(date +%T)] rgb_only.7z done"

echo "[$(date +%T)] streaming RealSense test tree (3.5 GB, no local temp)"
tar -C "$CW1" --exclude '.DS_Store' -cf - "Test data-COMP0248_Test_data_23" \
  | sshpass -P "assword:" -f $PASS ssh "${OPTS[@]}" "$REMOTE" "cat > $RAW/realsense_test.tar" || exit 1
echo "[$(date +%T)] test tree done"

echo "[$(date +%T)] remote verification"
sshpass -P "assword:" -f $PASS ssh "${OPTS[@]}" "$REMOTE" \
  "ls -l $RAW; md5sum $RAW/rgb_only.7z" || exit 1
echo "[$(date +%T)] UPLOAD_ALL_DONE"
