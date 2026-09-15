#!/bin/bash
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