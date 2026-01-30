"""
Plot evaluation behavior visualization for ICML paper.
Left panel: 2D navigation trajectories with gradient coloring by episode
Right panel: Returns per sub-trajectory with gradient coloring

Clean, publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pathlib import Path
import pickle
import argparse


# ============================================================================
# Configuration
# ============================================================================

# Environment parameters
RADIUS = 1.0
GOAL_TOLERANCE = 0.2
HORIZON = 20  # steps per episode
N_EPISODES = 40

# Visual settings - Blue gradient
CMAP_NAME = "Blues"


def load_eval_data(eval_dir: Path):
    """Load evaluation data from a directory."""
    # Load returns
    returns_path = eval_dir / "eval_returns.npz"
    returns_data = np.load(returns_path)
    
    # Load trajectories
    trajs_path = eval_dir / "eval_trajs.pkl"
    with open(trajs_path, 'rb') as f:
        trajs_data = pickle.load(f)
    
    return returns_data, trajs_data


def get_goal_positions(n_goals=100, radius=1.0, eval_only=False):
    """Generate goal positions on semicircle.
    
    If eval_only=True, returns only the 20% test/eval goals.
    """
    angles = np.linspace(0, np.pi, n_goals)
    goals = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
    
    if eval_only:
        # Shuffle with same seed as create_envs.py, then take last 20%
        np.random.RandomState(seed=0).shuffle(goals)
        split_idx = int(0.8 * len(goals))
        return goals[split_idx:]
    return goals


def compute_episode_returns(rewards, dones):
    """Compute returns for each episode from raw rewards and dones."""
    # rewards shape: (total_steps,)
    # Each episode has HORIZON=20 steps, so total_steps = 800 for 40 episodes
    episode_returns = []
    
    # Find done indices
    done_indices = np.where(dones)[0]
    
    # Split by episodes
    start = 0
    for end_idx in done_indices:
        ep_reward = rewards[start:end_idx+1].sum()
        episode_returns.append(ep_reward)
        start = end_idx + 1
    
    return np.array(episode_returns)


def plot_single_goal_behavior(states, rewards, dones, goal_idx, goal_pos, 
                               all_goals, output_path, title_suffix="", 
                               traj_steps=10, max_episodes=15):
    """
    Create a two-panel figure for a single goal:
    Left: Navigation behavior with gradient-colored trajectories (no ticks/labels)
    Right: Per-episode returns with gradient coloring (clean axis labels)
    
    Args:
        traj_steps: Number of steps to show per trajectory (default: 10)
        max_episodes: Maximum number of episodes to show (default: 15)
    """
    
    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    
    # Create figure with colorbar space - both plots same size rectangles
    fig = plt.figure(figsize=(9, 3.2), constrained_layout=True)
    
    # Create gridspec for proper layout with shared colorbar - equal sizes
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.15)
    ax_nav = fig.add_subplot(gs[0])
    ax_ret = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    
    # ========== Left panel: Navigation behavior (clean, no ticks) ==========
    ax_nav.set_facecolor('white')
    
    # Draw all goals as concentric circles (light gray fill, dashed border)
    for g in all_goals:
        circle = Circle(g, GOAL_TOLERANCE, 
                       facecolor='#e8e8e8',   # Light gray fill
                       edgecolor='#aaaaaa',   # Gray dashed border
                       linestyle='--',
                       linewidth=0.8,
                       alpha=0.8,
                       zorder=1)
        ax_nav.add_patch(circle)
    
    # Highlight current goal (light green fill)
    goal_circle = Circle(goal_pos, GOAL_TOLERANCE,
                        facecolor='#90EE90',  # Light green
                        edgecolor='#228B22',  # Forest green border
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=2)
    ax_nav.add_patch(goal_circle)
    
    # Add green star marker at goal center
    ax_nav.scatter(goal_pos[0], goal_pos[1], s=100, marker='*', 
                   color='#228B22', edgecolors='none', zorder=3)
    
    # Mark origin with yellow circle (black border)
    ax_nav.scatter(0, 0, s=70, marker='o', color='#FFD700', 
                   edgecolors='black', linewidth=1.2, zorder=100)
    
    # Parse episodes from states
    done_indices = np.where(dones)[0]
    n_episodes = min(len(done_indices), max_episodes)  # Limit to max_episodes
    done_indices = done_indices[:n_episodes]  # Only use first n episodes
    
    # Setup colormap for episodes (matching plot_iterative.py style)
    cmap = plt.cm.get_cmap(CMAP_NAME)
    norm = Normalize(vmin=0, vmax=n_episodes - 1)
    
    # Plot each episode trajectory (truncated to traj_steps)
    start_idx = 0
    for ep_idx, end_idx in enumerate(done_indices):
        # Extract trajectory for this episode
        traj = states[start_idx:end_idx+1]
        
        # Truncate to traj_steps
        traj = traj[:traj_steps]
        
        # Get color based on episode index (shift to avoid very light colors)
        color = cmap(0.1 + 0.9 * norm(ep_idx))
        
        # Plot trajectory line (style matching plot_iterative.py)
        ax_nav.plot(traj[:, 0], traj[:, 1], 
                   color=color, linewidth=1.8, alpha=0.95, zorder=3 + ep_idx)
        
        start_idx = end_idx + 1
    
    # Axis settings for navigation - CLEAN, NO TICKS
    ax_nav.set_xlim(-1.25, 1.25)
    ax_nav.set_ylim(-0.25, 1.25)
    ax_nav.set_aspect('equal')
    
    # Remove ticks and labels (clean modern look like ant.py)
    ax_nav.set_xticks([])
    ax_nav.set_yticks([])
    ax_nav.set_xlabel('')
    ax_nav.set_ylabel('')
    
    # No title
    
    # Clean thin border
    for spine in ['top', 'right', 'left', 'bottom']:
        ax_nav.spines[spine].set_visible(True)
        ax_nav.spines[spine].set_color('#333333')
        ax_nav.spines[spine].set_linewidth(0.8)
    
    # Add legend below behavior plot
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
               markeredgecolor='black', markersize=10, markeredgewidth=1.2, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#228B22',
               markeredgecolor='none', markersize=14, linestyle='None'),
    ]
    legend_labels = ['Initial Position', 'Hidden Goal']
    ax_nav.legend(legend_handles, legend_labels, loc='lower center',
                  bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fancybox=False,
                  shadow=False, edgecolor='#cccccc', framealpha=1.0, facecolor='white',
                  fontsize=11, handletextpad=0.3, columnspacing=1.0)
    
    # ========== Right panel: Returns per sub-trajectory ==========
    ax_ret.set_facecolor('white')
    
    # Compute per-episode returns (limited to n_episodes)
    all_episode_returns = compute_episode_returns(rewards, dones)
    episode_returns = all_episode_returns[:n_episodes]  # Limit to max_episodes
    episodes = np.arange(1, n_episodes + 1)
    
    # Plot connecting line (subtle)
    ax_ret.plot(episodes, episode_returns, '-', color='#666666', 
                linewidth=1.0, alpha=0.5, zorder=1)
    
    # Plot scatter points with gradient coloring (same color shift as trajectories)
    colors = [cmap(0.1 + 0.9 * norm(i)) for i in range(n_episodes)]
    scatter = ax_ret.scatter(episodes, episode_returns, 
                            c=colors, s=50, 
                            edgecolors='white', linewidth=0.7,
                            zorder=3)
    
    # Axis settings for returns
    ax_ret.set_xlabel('Test-Time Episodes', fontsize=13)
    ax_ret.set_ylabel('Episode Return', fontsize=13)
    # No title
    
    # Set y-axis limits with some padding
    y_min, y_max = episode_returns.min(), episode_returns.max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax_ret.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
    ax_ret.set_xlim(0, n_episodes + 1)
    
    # Clean spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax_ret.spines[spine].set_visible(True)
        ax_ret.spines[spine].set_color('#333333')
        ax_ret.spines[spine].set_linewidth(0.8)
    
    ax_ret.tick_params(axis='both', which='major', direction='out',
                       length=4, width=0.8, colors='#333333', labelsize=11)
    
    # Add colorbar to the right (with shifted colors matching the plot)
    # Create a custom colormap that matches the shifted colors we use
    from matplotlib.colors import LinearSegmentedColormap
    # Sample the original cmap at shifted positions
    shifted_colors = [cmap(0.1 + 0.9 * x) for x in np.linspace(0, 1, 256)]
    shifted_cmap = LinearSegmentedColormap.from_list('shifted_blues', shifted_colors)
    
    sm = ScalarMappable(cmap=shifted_cmap, norm=Normalize(vmin=1, vmax=n_episodes))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Test-Time Episodes', fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.8)
    
    # Save PNG
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save PDF for publication
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation behavior for ICML paper')
    parser.add_argument('--eval_dir', type=str, required=True,
                       help='Path to evaluation directory containing eval_returns.npz and eval_trajs.pkl')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as eval_dir)')
    parser.add_argument('--goal_idx', type=int, default=None,
                       help='Specific goal index to plot (default: plot a few representative ones)')
    parser.add_argument('--n_goals_plot', type=int, default=10,
                       help='Number of goals to plot if goal_idx not specified')
    parser.add_argument('--traj_steps', type=int, default=10,
                       help='Number of steps to show per trajectory (default: 10)')
    parser.add_argument('--max_episodes', type=int, default=40,
                       help='Maximum number of test-time episodes to show (default: 40)')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {eval_dir}...")
    returns_data, trajs_data = load_eval_data(eval_dir)
    
    # Extract data
    states = trajs_data['states']  # (n_goals, total_steps, 2)
    rewards = trajs_data['rewards']  # (n_goals, total_steps)
    dones = trajs_data['dones']  # (n_goals, total_steps)
    
    n_goals = states.shape[0]
    print(f"Found {n_goals} goals, each with {states.shape[1]} steps")
    
    # Get eval goal positions for visualization (~20 goals)
    all_goals = get_goal_positions(n_goals=100, radius=RADIUS, eval_only=True)
    print(f"Using {len(all_goals)} eval goals for visualization")
    
    # Determine which goals to plot
    if args.goal_idx is not None:
        goal_indices = [args.goal_idx]
    else:
        # Plot evenly spaced goals
        goal_indices = np.linspace(0, n_goals - 1, args.n_goals_plot, dtype=int)
    
    print(f"Generating {len(goal_indices)} plots...")
    
    # Infer actual goal positions from final trajectory positions
    # (since we don't have direct access to goal info)
    for goal_idx in goal_indices:
        # Get data for this goal
        goal_states = states[goal_idx]
        goal_rewards = rewards[goal_idx]
        goal_dones = dones[goal_idx]
        
        # Estimate goal position from where agent gets rewards
        # High reward positions should be near the goal
        reward_mask = goal_rewards > 0
        if reward_mask.sum() > 0:
            goal_pos = goal_states[reward_mask].mean(axis=0)
            # Snap to nearest goal on semicircle
            angles = np.arctan2(goal_pos[1], goal_pos[0])
            goal_pos = np.array([RADIUS * np.cos(angles), RADIUS * np.sin(angles)])
        else:
            # Fallback: use the last position of the last episode
            goal_pos = goal_states[-1]
        
        output_path = output_dir / f"behavior_goal_{goal_idx}.png"
        plot_single_goal_behavior(
            goal_states, goal_rewards, goal_dones, 
            goal_idx, goal_pos, all_goals, output_path,
            traj_steps=args.traj_steps,
            max_episodes=args.max_episodes
        )
    
    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

