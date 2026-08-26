#!/bin/zsh
# knuckles_run.sh <host-l> <local-script.sh> [--bg <jobname> | --tail <jobname> [n]]
#
# Runs a local bash script on a UCL Knuckles GPU host, non-interactively.
#
# Two facts drive the design:
#   1. The remote login shell is csh, so the command string handed to ssh is parsed by csh.
#      csh cannot parse `> f 2>&1`, so all bash-isms are kept inside uploaded scripts and the
#      csh-level command is limited to `mkdir`, `echo`, `base64`, `&&` and `bash <file>`.
#   2. Arbitrary quoting must survive two SSH hops, so scripts are base64-encoded in transit.
emulate -L zsh
setopt no_nomatch
host="$1"; script="$2"; mode="${3:-}"; job="${4:-job}"; n="${5:-80}"
[[ -z "$host" ]] && { echo "usage: knuckles_run.sh <host-l> <script.sh> [--bg <job>|--tail <job> [n]]" >&2; exit 2 }
source ~/.zshrc >/dev/null 2>&1
D='/var/tmp/cw1_$USER'

if [[ "$mode" == "--tail" ]]; then
  ssh-ucl-knuckles "$host" "tail -n $n $D/logs/${job}.log" 2>/dev/null
  exit $?
fi

[[ -z "$script" ]] && { echo "error: script required" >&2; exit 2 }
b64=$(base64 < "$script" | tr -d '\n')

# Small bash launcher, uploaded alongside, so csh never sees a bash redirect.
# Jobs run under `systemd-run --user` so they survive the SSH session ending
# (linger is enabled for the account, so the user manager persists after logout).
launcher='#!/usr/bin/env bash
# args: <dir> <jobname>   (dir is passed in already-expanded by the remote csh)
D="$1"; J="$2"
# Linger is a PER-HOST logind setting. Without it the systemd --user manager exits at logout and
# takes every transient unit with it: the job dies the instant the SSH session closes, leaving a
# zero-byte log and no error. Enabling it is idempotent and cheap, so do it on every launch
# rather than remembering which hosts have been used before.
loginctl enable-linger "$USER" >/dev/null 2>&1
mkdir -p "$D/logs" || { echo "MKDIR FAILED for $D/logs"; exit 1; }
LOG="$D/logs/$J.log"
: > "$LOG"
systemctl --user reset-failed "cw1-$J.service" 2>/dev/null
systemctl --user stop "cw1-$J.service" 2>/dev/null
if systemd-run --user --unit="cw1-$J" --collect --same-dir \
     --property=StandardOutput=append:"$LOG" \
     --property=StandardError=append:"$LOG" \
     bash "$D/$J.sh" >/dev/null 2>&1; then
  echo "LAUNCHED $J unit=cw1-$J log=$LOG"
else
  echo "systemd-run failed, falling back to setsid" | tee -a "$LOG"
  setsid nohup bash "$D/$J.sh" >> "$LOG" 2>&1 < /dev/null &
  echo "LAUNCHED $J (setsid) pid=$!"
fi'
lb64=$(printf '%s' "$launcher" | base64 | tr -d '\n')

if [[ "$mode" == "--bg" ]]; then
  ssh-ucl-knuckles "$host" \
    "mkdir -p $D/logs && echo $b64 | base64 -d > $D/${job}.sh && echo $lb64 | base64 -d > $D/_launch.sh && bash $D/_launch.sh $D ${job}" 2>/dev/null
else
  ssh-ucl-knuckles "$host" \
    "mkdir -p $D && echo $b64 | base64 -d > $D/_cmd.sh && bash $D/_cmd.sh" 2>/dev/null
fi
