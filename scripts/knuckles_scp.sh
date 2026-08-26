#!/bin/zsh
# knuckles_scp.sh up   <host-l> <local-path> <remote-path>
# knuckles_scp.sh down <host-l> <remote-path> <local-path>
# Same two-hop path as ssh-ucl-knuckles, but without -tt (a pty corrupts binary data).
dir="$1"; host="$2"; a="$3"; b="$4"
PASS=~/.ssh/.ucl_knuckles_pass
PROXY="ProxyCommand=sshpass -P \"assword:\" -f $PASS ssh -o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -W %h:%p ucl-knuckles"
OPTS=(-o PreferredAuthentications=password -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -o "$PROXY")
if [[ "$dir" == "up" ]]; then
  sshpass -P "assword:" -f $PASS scp -C "${OPTS[@]}" "$a" "ashihab@${host}.cs.ucl.ac.uk:$b"
elif [[ "$dir" == "down" ]]; then
  sshpass -P "assword:" -f $PASS scp -C "${OPTS[@]}" "ashihab@${host}.cs.ucl.ac.uk:$a" "$b"
elif [[ "$dir" == "rsync" ]]; then
  # rsync <host> <local-dir> <remote-dir>
  sshpass -P "assword:" -f $PASS rsync -az --delete --info=progress2 \
    -e "sshpass -P \"assword:\" -f $PASS ssh ${OPTS[*]}" "$a" "ashihab@${host}.cs.ucl.ac.uk:$b"
else
  echo "usage: knuckles_scp.sh up|down|rsync <host-l> <src> <dst>" >&2; exit 2
fi
