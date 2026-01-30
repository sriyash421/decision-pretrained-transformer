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
    "darkroom-hard",
    "keydoor-markovian",
    "keydoor-nonmarkovian",
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
    "navigation-episodic": 2,
    "navigation-nonepisodic": 2,
}


# ============================================================================
# Data Loading
# ============================================================================

def load_eval_returns(task: str, seed: str, iteration: int) -> dict:
    """Load eval_returns.npz for a specific task/seed/iteration."""
    subfolder = f"{task}-{seed}-{task}-{seed}"
    path = BASE_DIR / task / seed / subfolder / f"dagger_step_{iteration}" / "eval" / "eval_returns.npz"
    
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
    Create a 1 row x 5 column plot.
    Each subplot shows all iterations as separate lines.
    X-axis: test-time episodes
    Y-axis: return
    Line colors follow a gradient by iteration.
    """
    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',
    })
    
    n_tasks = len(TASKS)
    
    # Create figure: 1 row x 5 columns
    fig, axes = plt.subplots(1, n_tasks, figsize=(16, 3.2), 
                              gridspec_kw={'wspace': 0.3})
    
    # Colormap for iteration progression
    cmap = plt.cm.plasma
    
    # Normalized colorbar from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
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
        
        # Plot expert performance as black dotted line
        expert_val = EXPERT_PERFORMANCE[task]
        ax.axhline(y=expert_val, color='black', linestyle='--', linewidth=2, 
                   label='Expert', zorder=100)
        
        for iteration in range(1, n_iters + 1):
            data = results[task][iteration]
            if data["mean"] is None:
                continue
            
            mean = data["mean"]
            std_err = data["std_err"]
            
            # Normalize iteration to 0-1 range for this task
            normalized_iter = (iteration - 1) / (n_iters - 1) if n_iters > 1 else 0
            color = cmap(normalized_iter)
            
            # Plot line with error band
            ax.fill_between(episodes, mean - std_err, mean + std_err,
                           alpha=0.15, color=color, linewidth=0)
            ax.plot(episodes, mean, '-', color=color, linewidth=1.8,
                   label=f'Iter {iteration}', zorder=2 + iteration)
        
        # Styling
        ax.set_title(TASK_NAMES[task], fontweight='bold', pad=10)
        ax.set_xlabel('Test-time episodes')
        if idx == 0:
            ax.set_ylabel('Return')
        
        # Enhanced grid
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5, color='#cccccc', zorder=1)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3, color='#dddddd', zorder=1)
        ax.minorticks_on()
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set x limits
        ax.set_xlim(1, n_episodes)
    
    # Shared colorbar on the right (normalized 0 to 1)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Iteration', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    
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
