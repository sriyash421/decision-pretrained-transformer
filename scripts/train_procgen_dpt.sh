#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="$HOME/anaconda3/envs/hitl/bin/python"
CHECKPOINT="models/dpt_procgen_train16_eval100_H800_seed0.pt"

"$PYTHON" train.py \
  --env maze \
  --env_type procgen \
  --H 800 \
  --envs 50000 \
  --envs_eval 100 \
  --hists 1 \
  --samples 1 \
  --rollin_type uniform \
  --embd 256 \
  --head 4 \
  --layer 4 \
  --lr 1e-3 \
  --num_epochs 200 \
  --dropout 0.1 \
  --procgen_train_envs 1000 \
  --procgen_eval_envs 100 \
  --procgen_visibility 3 \
  --procgen_train_start 0 \
  --procgen_train_levels 1000 \
  --procgen_eval_start 1000 \
  --procgen_eval_levels 1000 \
  --seed 0

"$PYTHON" eval_procgen_dpt.py \
  --checkpoint "$CHECKPOINT" \
  --horizon 800 \
  --eval-horizon 800 \
  --n-embd 256 \
  --n-head 4 \
  --n-layer 4 \
  --dropout 0.1 \
  --seed 0 \
  --procgen-train-envs 16 \
  --procgen-eval-envs 100 \
  --procgen-train-start 0 \
  --procgen-train-levels 1000 \
  --procgen-eval-start 1000 \
  --procgen-eval-levels 1000 \
  --procgen-visibility 3
