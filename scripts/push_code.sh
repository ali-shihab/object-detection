#!/bin/zsh
# Ship the working tree (code + docs, no data) to the Knuckles workspace.
# macOS ships rsync 2.6.9 and quoting a ProxyCommand through `rsync -e` is fragile,
# so this tars, scps and untars instead. The tree is ~250 kB.
set -u
HOST="${1:-skate-l}"
L=~/workspace/projects/ucl/object_detection_and_classification/cw1/lsa
WS=/cs/student/project_msc/2025/rai/ashihab/comp0248_lsa
PASS=~/.ssh/.ucl_knuckles_pass
PROXY="ProxyCommand=sshpass -P \"assword:\" -f $PASS ssh -o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -W %h:%p ucl-knuckles"
OPTS=(-o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o "$PROXY")
REMOTE="ashihab@${HOST}.cs.ucl.ac.uk"
tar -C "$L" --exclude '.DS_Store' --exclude '__pycache__' --exclude '*.tgz' \
    -czf /tmp/lsa_code.tgz project_18006111_Shihab scripts docs report
sshpass -P "assword:" -f $PASS scp "${OPTS[@]}" /tmp/lsa_code.tgz "$REMOTE:/var/tmp/cw1_ashihab/lsa_code.tgz" || exit 1
sshpass -P "assword:" -f $PASS ssh "${OPTS[@]}" "$REMOTE" \
  "mkdir -p $WS/code && rm -rf $WS/code/tools $WS/code/tests $WS/code/configs && tar -xzf /var/tmp/cw1_ashihab/lsa_code.tgz -C $WS/code && find $WS/code -name '*.py' | wc -l" || exit 1
rm -f /tmp/lsa_code.tgz
echo "PUSH_CODE_DONE"
