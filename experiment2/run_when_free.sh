#!/bin/bash
# run_when_free.sh - waits for a GPU with enough free memory, pins to it, runs the command.
# Usage:  ./run_when_free.sh 20000 python -u score_validation.py
# First argument: required free MiB (20000 = an idle RTX 6000).
# Replaces CUDA_VISIBILITY.sh - do NOT source that script in the same shell.
THRESH=$1; shift
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -rn | awk -F', ' -v t="$THRESH" '$2 >= t {print $1; exit}')
  [ -n "$GPU" ] && break
  sleep 300
done
export CUDA_VISIBLE_DEVICES=$GPU
echo "$(date): starting on GPU $GPU"
exec "$@"