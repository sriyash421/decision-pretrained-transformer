"""
Plot test-time performance comparison across methods.
For ICML paper - clean publication-quality figures.
Version with markers and subsampled points.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import pickle

# ============================================================================
# Configuration
# ============================================================================

# Smoothing window for RL-based methods (set to 1 for no smoothing)
SMOOTH_WINDOW = 5

# Number of points to subsample for markers
SUBSAMPLE_POINTS = 8

BASE_DIR = Path("/checkpoint/siro/sriyash/dpt-code/results")
OUTPUT_DIR = Path("/checkpoint/siro/sriyash/dpt-code/plots")

TASKS = [
    "darkroom-easy",
    "darkroom-hard",
    "keydoor-markovian",
    "keydoor-nonmarkovian",
    "navigation-episodic",
    # "navigation-nonepisodic",
]

SEEDS = ["seed0", "seed1", "seed2"]

# Method configurations
METHODS = {
    "Ours": {
        "type": "context_accumulation",  # Special handling for our method
        "folder": "context_accumulation_final_runs_lower_data",
    },
    "DPT": {
        "type": "standard",
        # "folder": "dpt_final_runs",
        # "folder": "dpt_final_runs_tuned_params",
        "folder": "dpt_darkroom_epoch200_final",
        "prefix": "dpt",
        # "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_epoch500/eval_returns.npz",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_epoch200/eval_returns.npz",
    },
    "SPOC": {
        "type": "standard",
        "folder": "spoc_final_runs", 
        "prefix": "spoc",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval/eval_returns.npz",
    },
    "RL²": {
        "type": "standard",
        "folder": "rl2_aligned_final_no_norm",
        "prefix": "rl2_aligned_final_no_norm",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "VariBAD": {
        "type": "standard",
        "folder": "varibad_aligned",
        "prefix": "varibad",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "BC+RL²": {
        "type": "standard",
        "folder": "bcppo",
        "prefix": "bcppo",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "AAWR": {
        "type": "standard",
        "folder": "aawr_4_episodes",
        "prefix": "aawr",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval/eval_returns.npz",
    },
    "ADVISOR": {
        "type": "standard",
        "folder": "advisor",
        "prefix": "advisor",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
}

# Final iteration for each task (for context accumulation / Ours)
FINAL_ITERATIONS = {
    "darkroom-easy": 9,
    "darkroom-hard": 9,
    "keydoor-markovian": 9,
    "keydoor-nonmarkovian": 9,
    "navigation-episodic": 9,
    "navigation-nonepisodic": 9,
}

# Pretty names for plotting
TASK_NAMES = {
    "darkroom-easy": "Darkroom-Easy",
    "darkroom-hard": "Darkroom-Hard",
    "keydoor-markovian": "Keydoor-Markovian",
    "keydoor-nonmarkovian": "Keydoor-NonMarkovian",
    "navigation-episodic": "2D-Navigation",
    "navigation-nonepisodic": "2D-Navigation-Nonepisodic",
}

# Expert performance for each task
EXPERT_PERFORMANCE = {
    "darkroom-easy": 93,
    "darkroom-hard": 192,
    "keydoor-markovian": 95,
    "keydoor-nonmarkovian": 2,
    "navigation-episodic": 12,
    "navigation-nonepisodic": 20,
}

# Colors - Tableau palette with lighter/darker purple for RL² family
METHOD_COLORS = {
    "Ours": "#003366",             # Navy blue
    "DPT": "#2ca02c",              # Tableau green
    "SPOC": "#ff7f0e",             # Tableau orange
    "RL²": "#9467bd",              # Tableau purple (standard)
    "BC+RL²": "#6a3d9a",           # Darker purple
    "VariBAD": "#d62728",          # Tableau red
    "AAWR": "#8c564b",             # Tableau brown
    "ADVISOR": "#7f7f7f",          # Tableau gray
}

# Markers for each method
METHOD_MARKERS = {
    "Ours": "o",
    "DPT": "s",
    "SPOC": "^",
    "RL²": "D",
    "BC+RL²": "v",
    "VariBAD": "<",
    "AAWR": ">",
    "ADVISOR": "p",
}

# Methods that use RL (apply smoothing to these)
# RL_METHODS = ["RL²", "BC+RL²", "VariBAD", "AAWR", "ADVISOR"]

# METHOD_ORDER = ["Ours", "DPT", "SPOC", "RL²", "BC+RL²", "VariBAD", "AAWR", "ADVISOR"]

RL_METHODS = ["RL²", "BC+RL²", "AAWR", "ADVISOR"]

METHOD_ORDER = ["Ours", "DPT", "RL²", "BC+RL²", "ADVISOR", "AAWR", "SPOC"]



def smooth(data, window):
    """Apply sliding window smoothing."""
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges
    smoothed = np.convolve(data, kernel, mode='same')
    # Fix edge effects by not smoothing first/last few points
    half = window // 2
    smoothed[:half] = data[:half]
    smoothed[-half:] = data[-half:]
    return smoothed


# ============================================================================
# Data Loading
# ============================================================================

def load_eval_returns(method: str, task: str, seed: str) -> dict:
    """Load eval_returns.npz for a specific method/task/seed."""
    config = METHODS[method]
    
    if config["type"] == "context_accumulation":
        # Special path for our context accumulation method
        final_iter = FINAL_ITERATIONS[task]
        subfolder = f"{task}-{seed}-{task}-{seed}"
        path = BASE_DIR / config["folder"] / task / seed / subfolder / f"dagger_step_{final_iter}" / "eval" / "eval_returns.npz"
    else:
        # Standard path pattern
        path_template = config["path_pattern"]
        rel_path = path_template.format(
            folder=config["folder"],
            prefix=config["prefix"],
            task=task,
            seed=seed
        )
        path = BASE_DIR / rel_path
    
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    
    data = np.load(path)
    return {
        "mean_returns": data["mean_returns"],
        "std_returns": data["std_returns"],
    }


def aggregate_across_seeds(method: str, task: str) -> dict:
    """
    Aggregate results across seeds for a given method and task.
    Returns mean and std_err arrays across all test-time episodes.
    """
    all_means = []
    
    for seed in SEEDS:
        data = load_eval_returns(method, task, seed)
        if data is None:
            continue
        all_means.append(data["mean_returns"])
    
    if len(all_means) == 0:
        return {"mean": None, "std_err": None}
    
    all_means = np.array(all_means)
    n_seeds = all_means.shape[0]
    
    overall_mean = np.mean(all_means, axis=0)
    
    if n_seeds > 1:
        std_err = np.std(all_means, axis=0, ddof=1) / np.sqrt(n_seeds)
    else:
        std_err = np.zeros_like(overall_mean)
    
    return {"mean": overall_mean, "std_err": std_err}


def build_results_dict() -> dict:
    """
    Build the results dictionary.
    Returns: {method: {task: {"mean": array, "std_err": array}}}
    """
    results = {}
    
    for method in METHOD_ORDER:
        results[method] = {}
        print(f"\nLoading {method}...")
        
        for task in TASKS:
            agg = aggregate_across_seeds(method, task)
            agg["mean"] = agg["mean"][:40]
            agg["std_err"] = agg["std_err"][:40]
            results[method][task] = agg
            
            if agg["mean"] is not None:
                print(f"  {task}: {len(agg['mean'])} episodes, "
                      f"final return = {agg['mean'][-1]:.2f}")
            else:
                print(f"  {task}: No data")
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(results: dict, output_path: Path):
    """
    Create a 1 row x N column plot comparing all methods.
    Returns are normalized by expert performance.
    Shared legend at the bottom.
    Uses markers and subsampled points.
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
    
    # Square-ish subplots: width per subplot ~3.0, height ~2.8
    subplot_size = 2.8
    fig_width = n_tasks * subplot_size + 1.5  # Extra space for spacing and legend
    fig_height = subplot_size + 0.8  # Extra space for labels/legend
    
    # Create figure with space for legend at bottom
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height))
    
    # Handle single task case
    if n_tasks == 1:
        axes = [axes]
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        
        # Get number of episodes from first valid method
        n_episodes = None
        for method in METHOD_ORDER:
            data = results[method][task]
            if data["mean"] is not None:
                n_episodes = len(data["mean"])
                break
        
        if n_episodes is None:
            continue
        
        episodes = np.arange(1, n_episodes + 1)
        
        # Get expert performance for normalization
        expert_val = EXPERT_PERFORMANCE[task]
        
        # Plot expert performance as horizontal line at y=1.0 (normalized)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, zorder=100)
        
        # Compute subsample indices
        subsample_indices = np.linspace(0, n_episodes - 1, SUBSAMPLE_POINTS, dtype=int)
        
        # Plot each method (normalized by expert performance)
        for method in METHOD_ORDER:
            data = results[method][task]
            if data["mean"] is None:
                continue
            
            # Normalize by expert performance
            mean = data["mean"] / expert_val
            std_err = data["std_err"] / expert_val
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            
            # Apply smoothing to RL methods
            if method in RL_METHODS and SMOOTH_WINDOW > 1:
                mean = smooth(mean, SMOOTH_WINDOW)
                std_err = smooth(std_err, SMOOTH_WINDOW)
            
            # Subsample for markers
            episodes_sub = episodes[subsample_indices]
            mean_sub = mean[subsample_indices]
            std_err_sub = std_err[subsample_indices]
            
            # Plot with error band (paper style: soft shaded regions)
            ax.fill_between(episodes, mean - std_err, mean + std_err,
                           alpha=0.15, color=color, linewidth=0)
            
            # "Ours" on top with thicker line
            lw = 2.5 if method == "Ours" else 2
            zorder = 20 if method == "Ours" else 10
            
            # Plot line
            ax.plot(episodes, mean, '-', color=color, linewidth=lw,
                   zorder=zorder)
            
            # Plot markers at subsampled points
            ax.scatter(episodes_sub, mean_sub, 
                      s=40, 
                      marker=marker,
                      color=color,
                      edgecolors='white',
                      linewidth=1.0,
                      zorder=zorder + 1)
        
        # Styling (paper style - no bold title)
        ax.set_title(TASK_NAMES[task], pad=8)
        ax.set_xlabel('Test-Time Episodes')
        if idx == 0:
            ax.set_ylabel('Normalized Return')
        
        # Grid (paper style - visible grid lines)
        ax.grid(True, alpha=0.8, linestyle='-', linewidth=0.5, color='#b0b0b0', zorder=0)
        ax.set_axisbelow(True)  # Grid behind data
        
        # Full boundary/frame (all 4 spines visible)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('black')
        
        ax.set_xlim(1, n_episodes)
        
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(-0.1, 1.1)
        # Set aspect ratio to make plots more square
        ax.set_box_aspect(1)
    
    # Shared legend at the bottom (paper style: clean, simple, single row)
    # Create custom handles with both line and marker
    legend_handles = []
    legend_labels = []
    
    # Expert line (dashed, no marker)
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
    legend_labels.append('Expert')
    
    # Method handles with line + marker
    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        lw = 2.5 if method == "Ours" else 2
        handle = Line2D([0], [0], color=color, linestyle='-', linewidth=lw,
                       marker=marker, markersize=7, markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.0)
        legend_handles.append(handle)
        legend_labels.append(method)
    
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(METHOD_ORDER) + 1,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=False,
               shadow=False, edgecolor='#cccccc', framealpha=1.0,
               facecolor='white', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved comparison plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    results = build_results_dict()
    
    # Save dictionary
    dict_path = OUTPUT_DIR / "test_time_comparison_results_markers.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results dict to {dict_path}")
    
    # Create plot
    print("\nGenerating plot...")
    plot_comparison(results, OUTPUT_DIR / "test_time_comparison_markers.png")
    
    print("\nDone!")

