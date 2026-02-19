import numpy as np
from gymnasium import spaces

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import os


DX = [0, 0, -1, 1]
DY = [1, -1, 0, 0]
N_ACT = 4


def _render_grid_obs(obs_3ch, cell_px=40):
    wd = obs_3ch.shape[0]
    size = wd * cell_px
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for gy in range(wd):
        for gx in range(wd):
            py = gy * cell_px
            px = gx * cell_px
            ch0, ch1, ch2 = obs_3ch[gy, gx]
            if ch0 < -0.5:
                img[py:py + cell_px, px:px + cell_px] = 120
            elif ch0 > 0.5:
                pass
            else:
                img[py:py + cell_px, px:px + cell_px] = 255
            if ch1 > 0.5:
                img[py:py + cell_px, px:px + cell_px] = (0, 200, 0)
            if ch2 > 0.5:
                m = cell_px // 5
                img[py + m:py + cell_px - m,
                    px + m:px + cell_px - m] = (220, 40, 40)
    for k in range(wd + 1):
        coord = k * cell_px
        if coord < size:
            img[coord, :] = 80
            img[:, coord] = 80
    return img


def q_iteration(nav, goal, discount=0.99, num_itrs=200):
    wd = nav.shape[0]
    free = [(x, y) for y in range(wd) for x in range(wd) if nav[y, x]]
    idx_map = {c: i for i, c in enumerate(free)}
    ns = len(free)
    if ns == 0:
        return np.full((wd, wd), -1, np.int32)

    goal_idx = idx_map.get(goal, -1)
    next_state = np.zeros((ns, N_ACT), dtype=np.int32)
    reward = np.zeros((ns, N_ACT), dtype=np.float64)

    for si, (x, y) in enumerate(free):
        for a in range(N_ACT):
            nx, ny = x + DX[a], y + DY[a]
            next_state[si, a] = idx_map.get((nx, ny), si)
            if next_state[si, a] == goal_idx:
                reward[si, a] = 1.0

    q = np.zeros((ns, N_ACT), dtype=np.float64)
    for _ in range(num_itrs):
        v = np.max(q, axis=1)
        q = reward + discount * v[next_state]

    opt = np.full((wd, wd), -1, dtype=np.int32)
    for si, (x, y) in enumerate(free):
        opt[y, x] = int(np.argmax(q[si]))
    return opt


def load_maze_and_goals(maze_path, train_ratio=0.8, seed=42):
    nav = np.load(maze_path)
    wd = nav.shape[0]
    free = [(x, y) for y in range(wd) for x in range(wd) if nav[y, x]]
    np.random.seed(seed)
    # idxs = np.random.permutation(len(free))
    # n_train = int(len(free) * train_ratio)
    # train_goals = [free[i] for i in idxs[:n_train]]
    # eval_goals = [free[i] for i in idxs[n_train:]]
    train_goals = free
    eval_goals = free
    return nav, train_goals, eval_goals


class MazeEnv:
    def __init__(self, nav, goal_cells, visibility=7, max_steps=500):
        self.nav = nav
        self.wd = nav.shape[0]
        self.goal_cells = goal_cells
        self.visibility = min(visibility, self.wd) if visibility is not None else self.wd
        self.max_steps = max_steps
        self.free_cells = [(x, y) for y in range(self.wd)
                           for x in range(self.wd) if nav[y, x]]

        self.observation_space = spaces.Box(
            -1, 1, (self.visibility, self.visibility, 3), np.float32)
        self.action_space = spaces.Discrete(N_ACT)

        self.agent_pos = None
        self.goal_pos = None
        self.opt = None
        self.steps = 0

    def reset(self):
        self.goal_pos = self.goal_cells[np.random.randint(len(self.goal_cells))]
        free = [c for c in self.free_cells if c != self.goal_pos]
        self.agent_pos = free[np.random.randint(len(free))]
        self.opt = q_iteration(self.nav, self.goal_pos)
        self.steps = 0
        return self._obs(), self._info()

    def step(self, action):
        ax, ay = self.agent_pos
        nx, ny = ax + DX[action], ay + DY[action]
        if 0 <= nx < self.wd and 0 <= ny < self.wd and self.nav[ny, nx]:
            self.agent_pos = (nx, ny)

        success = self.agent_pos == self.goal_pos
        reward = 1.0 if success else 0.0
        self.steps += 1
        done = success or (self.steps >= self.max_steps)
        return self._obs(), reward, done, self._info()

    def _obs(self):
        v = self.visibility
        obs = np.full((v, v, 3), -1.0, dtype=np.float32)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        # Full-grid mode: visibility == world size, no agent-centered crop.
        if v == self.wd:
            obs[:, :, 0] = np.where(self.nav, 0.0, 1.0)
            obs[gy, gx, 1] = 1.0
            obs[ay, ax, 2] = 1.0
            return obs

        half = v // 2
        for oy in range(v):
            for ox in range(v):
                dx, dy = ox - half, oy - half
                wx, wy = ax + dx, ay + dy
                if 0 <= wx < self.wd and 0 <= wy < self.wd:
                    obs[oy, ox, 0] = 0.0 if self.nav[wy, wx] else 1.0
                    obs[oy, ox, 1] = 1.0 if (wx == gx and wy == gy) else 0.0
                    obs[oy, ox, 2] = 1.0 if (wx == ax and wy == ay) else 0.0
        # obs = _render_grid_obs(obs, cell_px=40)  # (v*40, v*40, 3) uint8
        return obs

    def _info(self):
        ax, ay = self.agent_pos
        opt_action = self.opt[ay, ax] if self.opt[ay, ax] >= 0 else 0
        full_obs = np.full((self.wd, self.wd, 3), -1.0, dtype=np.float32)
        full_obs[:, :, 0] = np.where(self.nav, 0.0, 1.0)
        gx, gy = self.goal_pos
        full_obs[gy, gx, 1] = 1.0
        full_obs[ay, ax, 2] = 1.0
        return {"opt_action": opt_action, "agent_pos": self.agent_pos,
                "goal_pos": self.goal_pos, "full_obs": full_obs}

    def expert_action(self):
        return self._info()["opt_action"]


class VecMazeEnv:
    def __init__(self, nav, goal_cells, n_envs, visibility=7, max_steps=500):
        self.envs = [MazeEnv(nav, goal_cells, visibility, max_steps)
                     for _ in range(n_envs)]
        self.n = n_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.visibility = visibility

    def reset(self):
        obs, infos = [], []
        for env in self.envs:
            o, i = env.reset()
            obs.append(o)
            infos.append(i)
        return np.stack(obs), infos

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, i = env.step(a)
            if d:
                o, i = env.reset()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(i)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def expert_actions(self):
        return np.array([env.expert_action() for env in self.envs])


def make_maze_envs(
    n_train=16,
    n_eval=100,
    visibility=7,
    train_goal_ratio=0.8,
    seed=42,
    maze_path="maze.npy",
    max_steps=500,
    **kwargs,
):
    del kwargs
    if visibility is None:
        visibility = 7
    nav, train_goals, eval_goals = load_maze_and_goals(
        maze_path, train_ratio=train_goal_ratio, seed=seed
    )
    train_env = VecMazeEnv(nav, train_goals, n_envs=n_train,
                           visibility=visibility, max_steps=max_steps)
    eval_env = VecMazeEnv(nav, eval_goals, n_envs=n_eval,
                          visibility=visibility, max_steps=max_steps)
    return train_env, eval_env


if __name__ == "__main__":
    nav, train_goals, eval_goals = load_maze_and_goals("maze.npy")
    print(f"maze shape: {nav.shape}, train goals: {len(train_goals)}, "
          f"eval goals: {len(eval_goals)}")

    env = VecMazeEnv(nav, train_goals, n_envs=4, visibility=7)
    obs, infos = env.reset()
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"agent positions: {[i['agent_pos'] for i in infos]}")
    print(f"goal positions: {[i['goal_pos'] for i in infos]}")

    total_reward = 0
    for t in range(100):
        actions = env.expert_actions()
        obs, rews, dones, infos = env.step(actions)
        total_reward += rews.sum()
    print(f"total reward over 100 steps: {total_reward}")

    # --- Trajectory video saving for a single agent ---
    print("Saving trajectory video for a single agent...")
    from matplotlib import colors
    single_env = MazeEnv(nav, train_goals, visibility=100)
    
    frames = []
    traj_len = 100
    for ep_idx in range(10):
        obs, info = single_env.reset()
        for t in range(traj_len):
            # Render full maze with agent and goal
            maze_img = np.ones((nav.shape[0], nav.shape[1], 3), dtype=np.float32)
            # Walls: black, Free: white
            maze_img[nav == 0] = [0, 0, 0]
            # Goal: green
            gx, gy = single_env.goal_pos
            maze_img[gy, gx] = [0, 1, 0]
            # Agent: red
            ax, ay = single_env.agent_pos
            maze_img[ay, ax] = [1, 0, 0]
            frames.append(maze_img.copy())
            action = single_env.expert_action()
            obs, reward, done, info = single_env.step(action)
            if done:
                break

    # Save as mp4 using matplotlib animation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    im = ax.imshow(frames[0], interpolation='nearest')

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save('maze_debug_videos/maze_trajectory.mp4', writer='ffmpeg', fps=10)
    plt.close(fig)
    print("Saved maze_trajectory.mp4")

    # --- Save trajectory as GIF and individual frames ---
    print("Saving trajectory GIF and frames...")
    env = MazeEnv(nav, train_goals, visibility=100)
    obs, info = env.reset()
    frames = []
    os.makedirs("maze_debug_videos", exist_ok=True)
    for t in range(30):
        action = env.expert_action()
        obs, reward, done, info = env.step(action)
        # Visualization: left = maze with agent/goal, right = obs layers
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        # Maze view
        maze_img = np.zeros((env.wd, env.wd, 3), dtype=np.uint8)
        for y in range(env.wd):
            for x in range(env.wd):
                maze_img[y, x] = [255, 255, 255] if nav[y, x] else [0, 0, 0]
        gx, gy = info['goal_pos']
        maze_img[gy, gx] = [0, 255, 0]
        ax, ay = info['agent_pos']
        maze_img[ay, ax] = [255, 0, 0]
        axes[0].imshow(maze_img[::-1])
        axes[0].set_title(f"Maze t={t}")
        axes[0].axis('off')
        # Obs layers
        axes[1].imshow(obs[:, :, 0], cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title('Wall')
        axes[1].axis('off')
        axes[2].imshow(obs[:, :, 1], cmap='Greens', vmin=-1, vmax=1)
        axes[2].set_title('Goal')
        axes[2].axis('off')
        axes[3].imshow(obs[:, :, 2], cmap='Reds', vmin=-1, vmax=1)
        axes[3].set_title('Agent')
        axes[3].axis('off')
        fig.tight_layout()
        # Save frame
        fname = f"/tmp/frame_{t:03d}.png"
        fig.savefig(fname)
        frames.append(imageio.imread(fname))
        plt.close(fig)
    # Save video
    imageio.mimsave("maze_debug_videos/trajectory.gif", frames, duration=0.1)
    print("Saved maze_debug_videos/trajectory.gif and individual frames.")
