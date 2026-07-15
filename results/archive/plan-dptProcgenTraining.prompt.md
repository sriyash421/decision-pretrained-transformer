## Plan: DPT Procgen Training Integration

Adapt the existing DPT path in `train.py` to support Procgen with minimal branching, reusing the darkroom training/eval structure. The plan keeps horizon/context behavior explicit for non-episodic Procgen, swaps large pre-initialized sampling for reset-based rollout collection, and plugs in simple-window observations through the same controller/eval interfaces so train/eval stays uniform.

### Steps
1. Add a Procgen mode gate in [`train.py`](/home/sriyash/Projects/decision-pretrained-transformer/train.py) near env/model config and `DPT` setup symbols.
2. Tie `horizon == context_length` for Procgen in [`train.py`](/home/sriyash/Projects/decision-pretrained-transformer/train.py) where sequence lengths are built.
3. Implement reset-per-collection Procgen collector in [`train.py`](/home/sriyash/Projects/decision-pretrained-transformer/train.py) (or nearest data helper used by `DPT`).
4. Reuse [`procgen_env.py`](/home/sriyash/Projects/decision-pretrained-transformer/procgen_env.py) `VecProcgenMaze` with `local_window_obs` for sampling and training observations.
5. Add a minimal “move agent to sampled location then read `simple_window_obs`” helper in [`procgen_env.py`](/home/sriyash/Projects/decision-pretrained-transformer/procgen_env.py), called only by collector path.
6. Route eval through existing darkroom-style loop with Procgen env counts (`n_train_envs=16`, `n_eval_envs=100`) in [`train.py`](/home/sriyash/Projects/decision-pretrained-transformer/train.py) and corresponding eval entrypoints.

### Further Considerations
1. **Sampling mode (confirmed):** use **exact teleport** to sampled locations when collecting random Procgen observations.
2. **Episode semantics (confirmed):** keep Procgen **non-episodic** for DPT collection/eval (context stream; do not depend on episodic resets).
3. **Integration style (confirmed):** keep a **single entrypoint** and implement Procgen as an in-place branch in `train.py` via `--env_type procgen` (Option A), with no new launcher script.
