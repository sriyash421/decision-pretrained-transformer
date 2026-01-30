"""
Plot iterative context accumulation results across tasks and iterations.
For ICML paper - clean publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from pathlib import Path
import pickle

# ============================================================================
# Configuration
# ============================================================================
import sys
exp_name = sys.argv[1]  # e.g., context_accumulation_final_runs
BASE_DIR = Path(f"/checkpoint/siro/sriyash/dpt-code/results/{exp_name}")
OUTPUT_DIR = Path(f"/checkpoint/siro/sriyash/dpt-code/plots/iterative-{exp_name}")

TASKS = [
    "darkroom-easy",
    # "darkroom-hard",
    # "keydoor-markovian",
    # "keydoor-nonmarkovian",
    "navigation-episodic",
    # "navigation-nonepisodic",  # No eval data available yet
]

SEEDS = ["seed0", "seed1", "seed2"]

# Task-specific iteration counts (0-indexed, so 7 means 0-6, 10 means 0-9)
TASK_ITERATIONS = {
    "darkroom-easy": 10,        # 0-6
    "darkroom-hard": 10,        # 0-6
    "keydoor-markovian": 10,   # 0-9
    "keydoor-nonmarkovian": 10,
    "navigation-episodic": 10,
    "navigation-nonepisodic": 10,
}

# Pretty names for plotting
TASK_NAMES = {
    "darkroom-easy": "Darkroom-Easy",
    "darkroom-hard": "Darkroom-Hard",
    "keydoor-markovian": "Keydoor-Markovian",
    "keydoor-nonmarkovian": "Keydoor-NonMarkovian",
    "navigation-episodic": "2D-Navigation",
    "navigation-nonepisodic": "2D-Navigation",
}

# Expert performance for each task
EXPERT_PERFORMANCE = {
    "darkroom-easy": 93,
    "darkroom-hard": 192,
    "keydoor-markovian": 95,
    "keydoor-nonmarkovian": 2,
    "navigation-episodic": 12,
    "navigation-nonepisodic": 12,
}


# ============================================================================
# Data Loading
# ============================================================================

def load_eval_returns(task: str, seed: str, iteration: int) -> dict:
    """Load eval_returns.npz for a specific task/seed/iteration."""
    subfolder = f"{task}-{seed}-{task}-{seed}"
    path = BASE_DIR / task / seed / subfolder / f"dagger_step_{iteration}" / "eval" / "eval_returns.npz"
    
    # Try alternative naming convention (with -new suffix) for navigation tasks
    if not path.exists():
        subfolder_new = f"{task}-new-{seed}-{task}-new-{seed}"
        path = BASE_DIR / task / seed / subfolder_new / f"dagger_step_{iteration}" / "eval" / "eval_returns.npz"
    
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    
    data = np.load(path)
    return {
        "episode_returns": data["episode_returns"],
        "mean_returns": data["mean_returns"],  # Shape: (n_episodes,)
        "std_returns": data["std_returns"],    # Shape: (n_episodes,)
    }


def aggregate_across_seeds_full(task: str, iteration: int) -> dict:
    """
    Aggregate full episode curves across seeds for a given task and iteration.
    
    Returns mean and std_err arrays across all test-time episodes.
    """
    all_means = []
    
    for seed in SEEDS:
        data = load_eval_returns(task, seed, iteration)
        if data is None:
            continue
        all_means.append(data["mean_returns"])
    
    if len(all_means) == 0:
        return {"mean": None, "std_err": None}
    
    all_means = np.array(all_means)  # Shape: (n_seeds, n_episodes)
    n_seeds = all_means.shape[0]
    
    # Mean across seeds
    overall_mean = np.mean(all_means, axis=0)
    
    # Standard error of the mean
    if n_seeds > 1:
        std_err = np.std(all_means, axis=0, ddof=1) / np.sqrt(n_seeds)
    else:
        std_err = np.zeros_like(overall_mean)
    
    return {"mean": overall_mean, "std_err": std_err}


def build_results_dict() -> dict:
    """
    Build the results dictionary with full episode curves.
    
    Returns:
        {task: {iteration: {"mean": array, "std_err": array}}}
    """
    results = {}
    
    for task in TASKS:
        n_iters = TASK_ITERATIONS[task]
        results[task] = {}
        
        for i in range(n_iters):
            agg = aggregate_across_seeds_full(task, i)
            results[task][i + 1] = agg  # 1-indexed
        
        print(f"{task}: {n_iters} iterations loaded")
        if results[task][1]["mean"] is not None:
            print(f"  Episodes per iteration: {len(results[task][1]['mean'])}")
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_line_chart(results: dict, output_path: Path):
    """
    Create a 1 row x N column plot.
    Each subplot shows all iterations as separate lines.
    X-axis: test-time episodes
    Y-axis: normalized return (by expert performance)
    Line colors follow a gradient by iteration.
    """
    # Publication-quality settings (matching paper style)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.color': '#cccccc',
    })
    
    n_tasks = len(TASKS)
    
    # Square-ish subplots: width per subplot ~2.8, height ~2.8
    subplot_size = 2.8
    fig_width = n_tasks * subplot_size + 1.0  # Extra space for colorbar
    fig_height = subplot_size + 0.8  # Extra space for labels
    
    # Create figure: 1 row x N columns
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height), 
                              gridspec_kw={'wspace': 0.3})
    
    # Handle single task case
    if n_tasks == 1:
        axes = [axes]
    
    # Colormap for iteration progression (blue gradient matching paper style)
    cmap = plt.cm.Blues
    
    # Get the maximum number of iterations across all tasks for consistent colorbar
    max_iters = max(TASK_ITERATIONS.values())
    
    # Colorbar based on actual iteration numbers (1 to max_iters)
    norm = mcolors.Normalize(vmin=1, vmax=max_iters)
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        n_iters = TASK_ITERATIONS[task]
        
        # Get number of episodes from first valid iteration
        n_episodes = None
        for iteration in range(1, n_iters + 1):
            data = results[task][iteration]
            if data["mean"] is not None:
                n_episodes = len(data["mean"])
                break
        
        if n_episodes is None:
            continue
        
        episodes = np.arange(1, n_episodes + 1)
        
        # Get expert performance for normalization
        expert_val = EXPERT_PERFORMANCE[task]
        
        # Plot expert performance as horizontal line at y=1.0 (normalized)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
                   label='Expert', zorder=100)
        
        for iteration in range(1, n_iters + 1):
            data = results[task][iteration]
            if data["mean"] is None:
                continue
            
            # Normalize by expert performance
            mean = data["mean"] / expert_val
            std_err = data["std_err"] / expert_val
            
            # Use iteration number directly for color (shift to avoid very light colors)
            color = cmap(0.3 + 0.7 * norm(iteration))
            
            # Plot line with error band (paper style: soft shaded regions)
            ax.fill_between(episodes, mean - std_err, mean + std_err,
                           alpha=0.2, color=color, linewidth=0)
            ax.plot(episodes, mean, '-', color=color, linewidth=1.8,
                   label=f'Iter {iteration}', zorder=2 + iteration)
        
        # Styling (paper style - no bold title)
        ax.set_title(TASK_NAMES[task], pad=8)
        ax.set_xlabel('Episodes')
        if idx == 0:
            ax.set_ylabel('Normalized Return')
        
        # Grid (paper style - visible grid lines)
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.5, color='#b0b0b0', zorder=0)
        ax.set_axisbelow(True)  # Grid behind data
        
        # Full boundary/frame (all 4 spines visible)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('black')
        
        # Set x limits
        ax.set_xlim(1, n_episodes)
        
        # Set aspect ratio to make plots more square
        ax.set_box_aspect(1)
    
    # Shared colorbar on the right (showing actual iteration numbers)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Iteration', fontsize=10)
    # Set ticks at integer iteration values
    tick_values = list(range(1, max_iters + 1))
    # If too many ticks, show a subset
    if max_iters > 10:
        tick_values = [1] + list(range(5, max_iters + 1, 5))
        if max_iters not in tick_values:
            tick_values.append(max_iters)
    cbar.set_ticks(tick_values)
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved line chart to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build results dictionary
    print("Loading data...")
    results = build_results_dict()
    
    # Save dictionary
    dict_path = OUTPUT_DIR / "iterative_results.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results dict to {dict_path}")
    
    # Create plot
    print("\nGenerating plot...")
    plot_line_chart(results, OUTPUT_DIR / "iterative_lines.png")
    
    print("\nDone!")
