import numpy as np
import struct
from gymnasium import spaces
from procgen import ProcgenGym3Env


# procgen discrete(15) -> our discrete(4): 0=up, 1=down, 2=left, 3=right
# procgen action layout (set_action_xy): vx = act/3 - 1, vy = act%3 - 1
#   1=LEFT(vx=-1), 3=DOWN(vy=-1), 5=UP(vy=+1), 7=RIGHT(vx=+1)
# procgen world: x=right, y=up  (rendering flips y)
# numpy grid[y, x]: y=0 is bottom of world (top of screen)
ACT_MAP = {0: 5, 1: 3, 2: 1, 3: 7}
N_ACT = 4
# action 0=up(vy+1), 1=down(vy-1), 2=left(vx-1), 3=right(vx+1)
DX = [0, 0, -1, 1]
DY = [1, -1, 0, 0]
WALL_OBJ = 51
GOAL_OBJ = 2


class _Buf:
    """Sequential reader for procgen binary state."""
    def __init__(self, data):
        self.d = data
        self.o = 0

    def i(self):
        v = struct.unpack_from('<i', self.d, self.o)[0]
        self.o += 4
        return v

    def f(self):
        v = struct.unpack_from('<f', self.d, self.o)[0]
        self.o += 4
        return v

    def s(self):
        n = self.i()
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def skip(self, n):
        self.o += n

    def rng(self):
        self.i()  # is_seeded
        self.s()  # mt19937 state string

    def entity(self):
        x = self.f(); y = self.f()  # x, y
        self.skip(4 * 4)              # vx, vy, rx, ry
        tp = self.i()                 # type
        self.skip(4 * 2)              # image_type, image_theme
        self.i()                      # render_z
        self.skip(4 * 2)              # will_erase, collides_with_entities
        self.skip(4 * 3)              # collision_margin, rotation, vrot
        self.skip(4 * 6)              # is_reflected, fire_time, spawn_time, life_time, expire_time, use_abs_coords
        self.skip(4 * 4)              # friction, smart_step, avoids_collisions, auto_erase
        self.skip(4 * 6)              # alpha, health, theta, grow_rate, alpha_decay, climber_spawn_x
        return x, y, tp


def _extract_grid(env, idx):
    """Parse procgen maze binary state to get grid, agent pos, goal pos."""
    b = _Buf(env.callmethod("get_state")[idx])

    # --- Game::serialize ---
    b.i()              # SERIALIZE_VERSION
    b.s()              # game_name
    for _ in range(9):
        b.i()          # paint_vel_info, use_generated_assets, use_monochrome_assets,
                       # restrict_themes, use_backgrounds, center_agent,
                       # debug_mode, distribution_mode, use_sequential_levels
    for _ in range(3):
        b.i()          # use_easy_jump, plain_assets, physics_mode
    b.i()              # grid_step
    for _ in range(4):
        b.i()          # level_seed_low, level_seed_high, game_type, game_n
    b.rng()            # level_seed_rand_gen
    b.rng()            # rand_gen
    b.f()              # step_data.reward
    b.i(); b.i()       # done, level_complete
    b.i(); b.i()       # action, timeout
    for _ in range(4):
        b.i()          # current_level_seed, prev_level_seed, episodes_remaining, episode_done
    b.i()              # last_reward_timer
    b.f()              # last_reward
    b.i()              # default_action
    b.i()              # fixed_asset_seed
    b.i()              # cur_time
    b.i()              # is_waiting_for_step

    # --- BasicAbstractGame::serialize ---
    b.i()              # grid_size
    n_ents = b.i()     # entities count
    agent_x, agent_y = 0, 0
    for _ in range(n_ents):
        ex, ey, tp = b.entity()
        if tp == 0:    # PLAYER (object-ids.h: PLAYER=0)
            agent_x, agent_y = int(ex), int(ey)

    b.i()              # use_procgen_background
    b.i()              # background_index
    b.f(); b.f()       # bg_tile_ratio, bg_pct_x
    b.f()              # char_dim
    b.i(); b.i(); b.i()  # last_move_action, move_action, special_action
    b.f(); b.f(); b.f()  # mixrate, maxspeed, max_jump
    b.f(); b.f(); b.f()  # action_vx, action_vy, action_vrot
    b.f(); b.f()       # center_x, center_y
    b.i(); b.i()       # random_agent_start, has_useful_vel_info
    b.i()              # step_rand_int
    b.rng()            # asset_rand_gen
    mw = b.i()         # main_width
    mh = b.i()         # main_height
    b.i()              # out_of_bounds_object
    for _ in range(6):
        b.f()          # unit, view_dim, x_off, y_off, visibility, min_visibility

    # --- grid ---
    gw = b.i()
    gh = b.i()
    gn = b.i()         # vector length
    grid = np.array([b.i() for _ in range(gn)], dtype=np.int32).reshape(gh, gw)

    # --- MazeGame::serialize ---
    maze_dim = b.i()
    world_dim = b.i()

    nav = (grid != WALL_OBJ).astype(np.int32)
    gy, gx = np.where(grid == GOAL_OBJ)
    goal = (int(gx[0]), int(gy[0])) if len(gx) > 0 else None

    return nav, (agent_x, agent_y), goal, world_dim


def _q_iteration(nav, goal, world_dim, discount=0.99, num_itrs=200):
    """Q-value iteration on the grid. Returns optimal action map.

    Following D4RL pointmaze q_iteration.py:
      Q(s,a) = R(s,a) + discount * sum_s' T(s,a,s') * V(s')
      V(s) = max_a Q(s,a)
      opt(s) = argmax_a Q(s,a)

    States are (x, y) cells where nav[y, x] == 1.
    Actions: 0=up(dy=+1), 1=down(dy=-1), 2=left(dx=-1), 3=right(dx=+1).
    Reward: +10 at goal cell, 0 elsewhere.
    Transitions: deterministic; hitting wall stays in place.
    """
    # enumerate free cells as state indices
    free = []
    idx_map = {}
    for y in range(world_dim):
        for x in range(world_dim):
            if nav[y, x]:
                idx_map[(x, y)] = len(free)
                free.append((x, y))
    ns = len(free)
    if ns == 0:
        return np.full((world_dim, world_dim), -1, np.int32)

    gx, gy = goal
    goal_idx = idx_map.get((gx, gy), -1)

    # build transition matrix T[s, a] -> s' and reward R[s, a]
    # deterministic so T is just a next-state lookup
    next_state = np.zeros((ns, N_ACT), dtype=np.int32)
    reward = np.zeros((ns, N_ACT), dtype=np.float64)

    for si, (x, y) in enumerate(free):
        for a in range(N_ACT):
            nx, ny = x + DX[a], y + DY[a]
            if (nx, ny) in idx_map:
                next_state[si, a] = idx_map[(nx, ny)]
            else:
                next_state[si, a] = si  # wall: stay in place
            # reward for landing on goal
            nsi = next_state[si, a]
            if nsi == goal_idx:
                reward[si, a] = 10.0

    # Q-value iteration
    q = np.zeros((ns, N_ACT), dtype=np.float64)
    for _ in range(num_itrs):
        v = np.max(q, axis=1)
        q = reward + discount * v[next_state]

    # extract optimal action per cell
    opt = np.full((world_dim, world_dim), -1, dtype=np.int32)
    for si, (x, y) in enumerate(free):
        opt[y, x] = int(np.argmax(q[si]))
    return opt


def _render_opt_grid(nav, opt, agent_pos, goal_pos,
                     world_dim, img_size=512):
    """Render optimal-action arrows on the maze grid.

    Walls = black, free cells = white, arrows show optimal action,
    agent = red circle, goal = green square.
    y=0 in grid is bottom of world, so we flip y for the image.
    """
    cell = img_size // world_dim
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # arrow offsets for drawing: (dx_pixel, dy_pixel) per action
    # action 0=up(world +y = screen -row), 1=down, 2=left, 3=right
    arrow_color = (50, 120, 220)
    goal_color = (0, 200, 0)
    agent_color = (220, 40, 40)

    for gy in range(world_dim):
        for gx in range(world_dim):
            # flip y: world y=0 -> image row = (wd-1)*cell
            iy = (world_dim - 1 - gy) * cell
            ix = gx * cell
            if not nav[gy, gx]:
                # wall -> black (already 0)
                continue
            # free cell -> white background
            img[iy:iy + cell, ix:ix + cell] = 255

            a = opt[gy, gx]
            if a < 0:
                continue
            # draw arrow in cell center
            cy = iy + cell // 2
            cx = ix + cell // 2
            hl = cell // 3  # half-length
            # pixel directions (image coords)
            # up in world = -row in image
            pdx = [0, 0, -1, 1]
            pdy = [-1, 1, 0, 0]
            ex = cx + pdx[a] * hl
            ey = cy + pdy[a] * hl
            # draw line
            _draw_line(img, cx, cy, ex, ey, arrow_color, 2)
            # draw arrowhead
            _draw_circle(img, ex, ey, 3, arrow_color)

    # draw goal
    if goal_pos is not None:
        gx, gy = goal_pos
        iy = (world_dim - 1 - gy) * cell
        ix = gx * cell
        m = cell // 5
        img[iy + m:iy + cell - m, ix + m:ix + cell - m] = goal_color

    # draw agent
    if agent_pos is not None:
        ax, ay = agent_pos
        iy = (world_dim - 1 - ay) * cell + cell // 2
        ix = ax * cell + cell // 2
        _draw_circle(img, ix, iy, cell // 3, agent_color)

    return img


def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
    """Bresenham line draw on numpy image."""
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for t in range(-thickness, thickness + 1):
            for s in range(-thickness, thickness + 1):
                py, px = y0 + t, x0 + s
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(img, cx, cy, r, color):
    """Draw filled circle on numpy image."""
    h, w = img.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                py, px = cy + dy, cx + dx
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color


class VecProcgenMaze:
    """Vectorized gymnasium-style wrapper. Each sub-env shares one ProcgenGym3Env."""

    def __init__(self, env, render_mode=None):
        self.env = env
        self.n = env.num
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = spaces.Discrete(N_ACT)
        self._opt = [None] * self.n
        self._apos = [None] * self.n
        self._gpos = [None] * self.n
        self._last_rgb = [None] * self.n
        self._nav = [None] * self.n
        self._wd = [None] * self.n

    def _solve_all(self):
        for i in range(self.n):
            nav, apos, gpos, wd = _extract_grid(self.env, i)
            self._nav[i] = nav
            self._apos[i] = apos
            self._gpos[i] = gpos
            self._wd[i] = wd
            if gpos is not None:
                self._opt[i] = _q_iteration(nav, gpos, wd)
            else:
                self._opt[i] = None

    def _solve_one(self, i):
        nav, apos, gpos, wd = _extract_grid(self.env, i)
        self._nav[i] = nav
        self._apos[i] = apos
        self._gpos[i] = gpos
        self._wd[i] = wd
        if gpos is not None:
            self._opt[i] = _q_iteration(nav, gpos, wd)
        else:
            self._opt[i] = None

    def reset(self):
        _, ob, _ = self.env.observe()
        self._solve_all()
        obs = ob["rgb"]
        infos = self._build_infos()
        return obs, infos

    def step(self, actions):
        pa = np.array(
            [ACT_MAP[int(a)] for a in actions], dtype=np.int32
        )
        self.env.act(pa)
        rew, ob, first = self.env.observe()

        obs = ob["rgb"]
        rews = rew.astype(np.float32)
        dones = first.astype(bool)

        # re-extract agent positions from true binary state
        for i in range(self.n):
            if dones[i]:
                self._solve_one(i)
            else:
                self._refresh_apos(i)

        infos = self._build_infos()
        return obs, rews, dones, infos

    def _refresh_apos(self, i):
        """Re-read agent position from binary state."""
        nav, apos, _, _ = _extract_grid(self.env, i)
        self._apos[i] = apos

    def _build_infos(self):
        raw = self.env.get_info()
        infos = []
        for i in range(self.n):
            d = {}
            if self.render_mode == "rgb_array" and "rgb" in raw[i]:
                d["rgb"] = raw[i]["rgb"]
                self._last_rgb[i] = raw[i]["rgb"]
            if self._opt[i] is not None and self._apos[i] is not None:
                ax, ay = self._apos[i]
                oa = self._opt[i][ay, ax]
                d["opt_action"] = int(oa) if oa >= 0 else 0
            if self._nav[i] is not None and self._opt[i] is not None:
                d["opt_grid"] = _render_opt_grid(
                    self._nav[i], self._opt[i],
                    self._apos[i], self._gpos[i],
                    self._wd[i],
                )
            infos.append(d)
        return infos

    def render(self, idx=0):
        if self._last_rgb[idx] is not None:
            return self._last_rgb[idx]
        _, ob, _ = self.env.observe()
        return ob["rgb"][idx]


def make_maze_envs(
    n_train=8,
    n_eval=20,
    train_start=0,
    train_levels=100,
    eval_start=100,
    eval_levels=20,
):
    """Create train and eval ProcgenMaze envs wrapped in gymnasium style."""
    train_g3 = ProcgenGym3Env(
        num=n_train,
        env_name="maze",
        start_level=train_start,
        num_levels=train_levels,
        distribution_mode="easy",
        center_agent=True,
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=True,
    )
    eval_g3 = ProcgenGym3Env(
        num=n_eval,
        env_name="maze",
        start_level=eval_start,
        num_levels=eval_levels,
        distribution_mode="easy",
        center_agent=True,
        restrict_themes=True,
        use_backgrounds=False,
        render_mode="rgb_array",
        use_monochrome_assets=True,
    )
    train_env = VecProcgenMaze(train_g3)
    eval_env = VecProcgenMaze(eval_g3, render_mode="rgb_array")
    return train_env, eval_env


def _rollout_multi(env, policy, n_episodes=5, max_t=500):
    """Run rollouts collecting episodes from all sub-envs.

    Returns dict mapping (env_idx, episode_number) to
    list of (obs, rgb, opt_grid) frames.
    """
    obs, infos = env.reset()
    ep_count = [0] * env.n
    cur_frames = [[] for _ in range(env.n)]
    all_episodes = {}

    for t in range(max_t * n_episodes):
        acts = policy(obs, infos, env.n)
        for i in range(env.n):
            rgb = infos[i].get("rgb", obs[i])
            og = infos[i].get("opt_grid", None)
            cur_frames[i].append(
                (obs[i].copy(), rgb.copy(),
                 og.copy() if og is not None else None)
            )
        obs, rews, dones, infos = env.step(acts)
        for i in range(env.n):
            if dones[i]:
                if ep_count[i] < n_episodes:
                    all_episodes[(i, ep_count[i])] = cur_frames[i]
                ep_count[i] += 1
                cur_frames[i] = []
        if all(c >= n_episodes for c in ep_count):
            break
    for i in range(env.n):
        if ep_count[i] < n_episodes and cur_frames[i]:
            all_episodes[(i, ep_count[i])] = cur_frames[i]
    return all_episodes


def _save_video(frames, path, fps=10):
    """Save list of (obs, rgb, opt_grid) as side-by-side mp4."""
    import imageio_ffmpeg
    from PIL import Image
    combined = []
    for obs_f, rgb_f, og_f in frames:
        rh, rw = rgb_f.shape[:2]
        # resize obs to match rgb height
        obs_r = np.array(
            Image.fromarray(obs_f).resize(
                (rw, rh), Image.NEAREST
            )
        )
        panels = [obs_r, rgb_f]
        if og_f is not None:
            og_r = np.array(
                Image.fromarray(og_f).resize(
                    (rw, rh), Image.NEAREST
                )
            )
            panels.append(og_r)
        combined.append(np.concatenate(panels, axis=1))
    h, w = combined[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        path, (w, h), fps=fps, pix_fmt_in="rgb24"
    )
    writer.send(None)
    for frame in combined:
        writer.send(frame.tobytes())
    writer.close()
    print(f"saved {path} ({len(frames)} frames)")


if __name__ == "__main__":
    import os
    from PIL import Image

    out_dir = "maze_debug_videos"
    os.makedirs(out_dir, exist_ok=True)

    n_envs = 4
    n_episodes = 3
    _, eval_env = make_maze_envs(
        n_train=2, n_eval=n_envs,
        train_levels=10, eval_levels=20,
    )

    obs, infos = eval_env.reset()
    print(f"obs shape: {obs.shape}, obs dtype: {obs.dtype}")
    print(f"info keys: {list(infos[0].keys())}")
    if "rgb" in infos[0]:
        print(f"info rgb shape: {infos[0]['rgb'].shape}")
    print(f"opt_action: {infos[0].get('opt_action', 'N/A')}")

    # save static opt_grid images for each env
    for i in range(n_envs):
        og = infos[i].get("opt_grid")
        if og is not None:
            p = os.path.join(out_dir, f"opt_grid_env{i}.png")
            Image.fromarray(og).save(p)
            print(f"saved {p}")
        # print agent/goal positions for debugging
        print(f"  env{i}: agent={eval_env._apos[i]}, "
              f"goal={eval_env._gpos[i]}, "
              f"world_dim={eval_env._wd[i]}")

    # --- expert rollouts ---
    def expert_policy(obs, infos, n):
        return np.array(
            [inf.get("opt_action", 0) for inf in infos]
        )

    print(f"\nrunning expert rollouts ({n_episodes} eps "
          f"x {n_envs} envs)...")
    episodes = _rollout_multi(
        eval_env, expert_policy, n_episodes=n_episodes
    )
    for (ei, ep), frames in sorted(episodes.items()):
        path = os.path.join(
            out_dir, f"expert_env{ei}_ep{ep}.mp4"
        )
        _save_video(frames, path)

    # --- random rollouts ---
    def random_policy(obs, infos, n):
        return np.array(
            [np.random.randint(N_ACT) for _ in range(n)]
        )

    print(f"\nrunning random rollouts ({n_episodes} eps "
          f"x {n_envs} envs)...")
    episodes = _rollout_multi(
        eval_env, random_policy, n_episodes=n_episodes
    )
    for (ei, ep), frames in sorted(episodes.items()):
        path = os.path.join(
            out_dir, f"random_env{ei}_ep{ep}.mp4"
        )
        _save_video(frames, path)

    print(f"\ndone. videos saved to {out_dir}/")
