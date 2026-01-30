"""
Plot test-time performance comparison for darkroom-mini environment.
Single-task plot with markers.
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

TASK = "darkroom-mini"

SEEDS = ["seed0", "seed1", "seed2"]

# Method configurations for darkroom-mini
METHODS = {
    "Ours": {
        "type": "context_accumulation",
        "folder": "context_accumulation_final_runs_lower_data",
        "final_iter": 4,  # darkroom-mini goes up to dagger_step_4
    },
    "RL²": {
        "type": "standard",
        "folder": "rl2_aligned",
        "prefix": "rl2-aligned",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "ADVISOR": {
        "type": "standard",
        "folder": "advisor_runs_final",
        "prefix": "advisor",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "VariBAD": {
        "type": "standard",
        "folder": "varibad_runs",
        "prefix": "varibad",
        "path_pattern": "{folder}/{prefix}-{task}-{seed}/eval_returns.npz",
    },
    "AAWR": {
        "type": "context_accumulation",  # Uses the same structure as Ours
        "folder": "aawr_context",
        "final_iter": 0,  # Only has dagger_step_0
    },
}

# Expert performance for darkroom-mini (you may need to adjust this)
# Based on darkroom-easy being 93, mini might be similar or different
EXPERT_PERFORMANCE = 27  # Adjust based on your knowledge of the environment

# Colors - matching the main paper style
METHOD_COLORS = {
    "Ours": "#003366",             # Navy blue
    "RL²": "#9467bd",              # Tableau purple
    "VariBAD": "#d62728",          # Tableau red
    "AAWR": "#8c564b",             # Tableau brown
    "ADVISOR": "#7f7f7f",          # Tableau gray
}

# Markers for each method
METHOD_MARKERS = {
    "Ours": "o",
    "RL²": "D",
    "VariBAD": "<",
    "AAWR": ">",
    "ADVISOR": "p",
}

# Methods that use RL (apply smoothing to these)
RL_METHODS = ["RL²", "ADVISOR"]

# Order for plotting (Ours first)
METHOD_ORDER = ["Ours", "RL²", "ADVISOR"]


def smooth(data, window):
    """Apply sliding window smoothing."""
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    half = window // 2
    smoothed[:half] = data[:half]
    smoothed[-half:] = data[-half:]
    return smoothed


# ============================================================================
# Data Loading
# ============================================================================

def load_eval_returns(method: str, seed: str) -> dict:
    """Load eval_returns.npz for a specific method/seed."""
    config = METHODS[method]
    
    if config["type"] == "context_accumulation":
        # Special path for context accumulation methods
        final_iter = config["final_iter"]
        subfolder = f"{TASK}-{seed}-{TASK}-{seed}"
        path = BASE_DIR / config["folder"] / TASK / seed / subfolder / f"dagger_step_{final_iter}" / "eval" / "eval_returns.npz"
    else:
        # Standard path pattern
        path_template = config["path_pattern"]
        rel_path = path_template.format(
            folder=config["folder"],
            prefix=config["prefix"],
            task=TASK,
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


def aggregate_across_seeds(method: str) -> dict:
    """
    Aggregate results across seeds for a given method.
    Returns mean and std_err arrays across all test-time episodes.
    """
    all_means = []
    
    for seed in SEEDS:
        data = load_eval_returns(method, seed)
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
    Returns: {method: {"mean": array, "std_err": array}}
    """
    results = {}
    
    for method in METHOD_ORDER:
        print(f"\nLoading {method}...")
        agg = aggregate_across_seeds(method)
        
        if agg["mean"] is not None:
            # Truncate to 40 episodes
            agg["mean"] = agg["mean"][:40]
            agg["std_err"] = agg["std_err"][:40]
            print(f"  {len(agg['mean'])} episodes, final return = {agg['mean'][-1]:.2f}")
        else:
            print(f"  No data")
        
        results[method] = agg
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_single_task(results: dict, output_path: Path):
    """
    Create a single plot for darkroom-mini.
    Returns are normalized by expert performance.
    """
    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
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
    
    # Single square-ish plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    
    # Get number of episodes from first valid method
    n_episodes = None
    for method in METHOD_ORDER:
        data = results[method]
        if data["mean"] is not None:
            n_episodes = len(data["mean"])
            break
    
    if n_episodes is None:
        print("No data available!")
        return
    
    episodes = np.arange(1, n_episodes + 1)
    
    # Expert performance for normalization
    expert_val = EXPERT_PERFORMANCE
    
    # Plot expert performance as horizontal line at y=1.0 (normalized)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, zorder=100)
    
    # Compute subsample indices
    subsample_indices = np.linspace(0, n_episodes - 1, SUBSAMPLE_POINTS, dtype=int)
    
    # Plot each method (normalized by expert performance)
    for method in METHOD_ORDER:
        data = results[method]
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
        
        # Plot with error band
        ax.fill_between(episodes, mean - std_err, mean + std_err,
                       alpha=0.15, color=color, linewidth=0)
        
        # "Ours" on top with thicker line
        lw = 2.5 if method == "Ours" else 2
        zorder = 20 if method == "Ours" else 10
        
        # Plot line
        ax.plot(episodes, mean, '-', color=color, linewidth=lw, zorder=zorder)
        
        # Plot markers at subsampled points
        ax.scatter(episodes_sub, mean_sub,
                  s=50,
                  marker=marker,
                  color=color,
                  edgecolors='white',
                  linewidth=1.0,
                  zorder=zorder + 1)
    
    # Styling
    ax.set_title('Darkroom-Mini', pad=10)
    ax.set_xlabel('Test-Time Episodes')
    ax.set_ylabel('Normalized Return')
    
    # Grid
    ax.grid(True, alpha=0.8, linestyle='-', linewidth=0.5, color='#b0b0b0', zorder=0)
    ax.set_axisbelow(True)
    
    # Full boundary/frame
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')
    
    ax.set_xlim(1, n_episodes)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.1, 1.15)
    
    # Legend with line + marker
    legend_handles = []
    legend_labels = []
    
    # Expert line
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
    legend_labels.append('Expert')
    
    # Method handles
    for method in METHOD_ORDER:
        if results[method]["mean"] is None:
            continue
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        lw = 2.5 if method == "Ours" else 2
        handle = Line2D([0], [0], color=color, linestyle='-', linewidth=lw,
                       marker=marker, markersize=7, markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.0)
        legend_handles.append(handle)
        legend_labels.append(method)
    
    # Legend below the plot
    fig.legend(legend_handles, legend_labels, loc='lower center', 
               ncol=len(legend_labels),
               bbox_to_anchor=(0.5, -0.02),
               frameon=True, fancybox=False, shadow=False,
               edgecolor='#cccccc', framealpha=1.0, facecolor='white')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for darkroom-mini...")
    results = build_results_dict()
    
    # Save dictionary
    dict_path = OUTPUT_DIR / "darkroom_mini_results.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results dict to {dict_path}")
    
    # Create plot
    print("\nGenerating plot...")
    plot_single_task(results, OUTPUT_DIR / "darkroom_mini_comparison.png")
    
    print("\nDone!")

