#!/usr/bin/env bash
set -euo pipefail

cd /home/sriyash/Projects/decision-pretrained-transformer

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hitl

python - <<'PY'
import numpy as np
import torch
from procgen_env import make_maze_envs
from models import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H = 800
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.1
seed = 0

_, eval_env = make_maze_envs(
    n_train=16,
    n_eval=100,
    train_start=0,
    train_levels=1000,
    eval_start=1000,
    eval_levels=1000,
    visibility=3,
    local_window_obs=True,
)

obs, _ = eval_env.reset()
state_dim = int(np.prod(obs.shape[1:]))
action_dim = eval_env.action_space.n

cfg = {
    "horizon": H,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "n_layer": n_layer,
    "n_embd": n_embd,
    "n_head": n_head,
    "shuffle": False,
    "dropout": dropout,
    "test": True,
    "store_gpu": True,
    "continuous_action": False,
    "rollin_type": "uniform",
}

ckpt = f"models/dpt_procgen_train16_eval100_H800_seed{seed}.pt"
model = Transformer(cfg).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

N = 2000
correct = 0
total = 0

for _ in range(max(1, N // eval_env.n)):
    for i in range(eval_env.n):
        nav = eval_env._nav[i]
        ys, xs = np.where(nav > 0)
        j = np.random.randint(len(xs))
        eval_env._apos[i] = (int(xs[j]), int(ys[j]))

    q_obs = eval_env._build_grid_obs().reshape(eval_env.n, -1).astype(np.float32)
    q_infos = eval_env._build_infos(raw_rgb=None)
    gt = np.array([d.get("opt_action", 0) for d in q_infos], dtype=np.int64)

    b = eval_env.n
    batch = {
        "query_states": torch.from_numpy(q_obs).float().to(device),
        "context_states": torch.zeros((b, H, state_dim), dtype=torch.float32, device=device),
        "context_actions": torch.zeros((b, H, action_dim), dtype=torch.float32, device=device),
        "context_next_states": torch.zeros((b, H, state_dim), dtype=torch.float32, device=device),
        "context_rewards": torch.zeros((b, H, 1), dtype=torch.float32, device=device),
        "zeros": torch.zeros((b, state_dim**2 + action_dim + 1), dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        logits = model(batch)[:, -1, :]
    pred = logits.argmax(dim=-1).cpu().numpy()

    correct += (pred == gt).sum()
    total += b

print(f"Eval query accuracy: {correct/total:.4f} ({correct}/{total})")
PY
