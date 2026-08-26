#!/bin/zsh
# Bring results, figures, generated tables and training logs back to the local project.
# Deliverable artefacts must live on the Mac, not only on a GPU host whose /var/tmp is wiped.
set -u
HOST="${1:-skate-l}"
L=~/workspace/projects/ucl/object_detection_and_classification/cw1/lsa
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
TMP=/var/tmp/cw1_ashihab
PASS=~/.ssh/.ucl_knuckles_pass
PROXY="ProxyCommand=sshpass -P \"assword:\" -f $PASS ssh -o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -W %h:%p ucl-knuckles"
OPTS=(-o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o "$PROXY")
REMOTE="ashihab@${HOST}.cs.ucl.ac.uk"

# The pack command is built remotely by a tiny script, so no shell metacharacter has to survive
# two layers of quoting (the login shell on the far side is csh).
PACK='cd '"$WS"' && find runs -name "log.jsonl" -o -name "config.json" > '"$TMP"'/_logs.txt 2>/dev/null; tar -czf '"$TMP"'/artifacts.tgz results artifacts -T '"$TMP"'/_logs.txt 2>/dev/null; ls -l '"$TMP"'/artifacts.tgz'
B64=$(printf '%s' "$PACK" | base64 | tr -d '\n')
sshpass -P "assword:" -f $PASS ssh "${OPTS[@]}" "$REMOTE" \
  "echo $B64 | base64 -d > $TMP/_pack.sh && bash $TMP/_pack.sh" || exit 1
sshpass -P "assword:" -f $PASS scp "${OPTS[@]}" "$REMOTE:$TMP/artifacts.tgz" /tmp/artifacts.tgz || exit 1

mkdir -p "$L/project_18006111_Shihab/results"
tar -xzf /tmp/artifacts.tgz -C "$L/project_18006111_Shihab/results" && rm -f /tmp/artifacts.tgz
mkdir -p "$L/report/figures" "$L/report/generated"
cp -f "$L/project_18006111_Shihab/results/artifacts/figures/"*.pdf "$L/report/figures/" 2>/dev/null
cp -f "$L/project_18006111_Shihab/results/artifacts/figures/"*.png "$L/report/figures/" 2>/dev/null
cp -f "$L/project_18006111_Shihab/results/artifacts/generated/"*.tex "$L/report/generated/" 2>/dev/null
echo "FETCH_DONE"
du -sh "$L/project_18006111_Shihab/results" 2>/dev/null
