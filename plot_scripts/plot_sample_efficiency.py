"""
Plot sample efficiency comparison across methods.
For ICML paper - clean publication-quality figures.
Shows Environment Steps vs Success Rate (log scale x-axis).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("/checkpoint/siro/sriyash/dpt-code/plots")

# Tasks to plot
TASK_NAMES = [
    "Tactile-Cube-Pick",
    "Wrist-Peg-Insertion",
    # Add more tasks here later
]

# Number of subsampled points for markers
SUBSAMPLE_POINTS = 8

# Number of points for log-space resampled curve (for RL methods)
LOG_RESAMPLE_POINTS = 500

# Smoothing window for RL curves (in log-space resampled points)
RL_SMOOTH_WINDOW = 5

# If True, auto-adjust x-axis lower limit to just before first "Ours" data point
AUTO_XLIM_TO_DATA = False

# ============================================================================
# Task-specific data
# ============================================================================

# For each task, define:
#   - "expert": (mean, std) - will be plotted as horizontal line
#   - "Ours": list of (mean, std, cumulative_steps) tuples per iteration
#   - Other offline methods: (mean, std, steps) single tuple
#   - RL methods: list of wandb run URLs (will fetch learning curves)

TASK_DATA = {
    "Tactile-Cube-Pick": {
        "expert": (0.96, 0.0),
        "Ours": [
            # (mean, std, steps_this_iteration) - steps will be cumsum'd
            (0.44271, 0.08, 145718),
            (0.74479, 0.036084, 245145),
            (0.822292, 0.019042, 405405),
            (0.88021, 0.023868, 505451),
            (0.92188, 0.056337 , 605729)
        ],
        "BC": (0.5625, 0.095043 , 145604),  # (mean, std, steps)
        "SPOC": (0.52083, 0.073841, 145604),  # Using same steps as BC for comparison
        "DPT": (0.38583, 0.036084, 145604),
        "AAWR": (0.55729, 0.032526, 145504),
        "BC+PPO": [
            # List of wandb URLs - will fetch learning curves
            "sriyash-uw-team/incontext_exploration/runs/icfunyk1",
            "sriyash-uw-team/incontext_exploration/runs/1vlig9r7",
            "sriyash-uw-team/incontext_exploration/runs/bs7rj9zq",
        ],
        "ADVISOR": [
            "sriyash-uw-team/isaaclab/runs/0ldrg5g0",
            "sriyash-uw-team/isaaclab/runs/6sw6wxzy"
        ]
    },
    # Add more tasks here following the same structure
}
TASK_DATA["Wrist-Peg-Insertion"] = TASK_DATA["Tactile-Cube-Pick"]  # Placeholder for now

# Task-specific parameters
TASK_PARAMS = {
    "Tactile-Cube-Pick": {
        "metric": "Metrics/task_0_success_rate",  # wandb metric key for RL methods
        "steps_per_update": 65536,  # env steps per wandb log step
        "n_seeds": 3,  # number of seeds for stderr calculation
        "ylabel": "Success Rate",
        "xlim": (1e4, 1e8),
        "ylim": (-0.05, 1.05),
    },
}
TASK_PARAMS["Wrist-Peg-Insertion"] = TASK_PARAMS["Tactile-Cube-Pick"]  # Placeholder

# ============================================================================
# Colors and markers - matching plot_test_time_performance.py
# ============================================================================

METHOD_COLORS = {
    "Ours": "#003366",             # Navy blue
    "DPT": "#2ca02c",              # Tableau green
    "SPOC": "#ff7f0e",             # Tableau orange
    "BC+PPO": "#6a3d9a",           # Darker purple
    "VariBAD": "#d62728",          # Tableau red
    "AAWR": "#8c564b",             # Tableau brown
    "ADVISOR": "#7f7f7f",          # Tableau gray
    "BC": "#1f77b4",               # Tableau blue
    "expert": "#000000",           # Black for expert
}

METHOD_MARKERS = {
    "Ours": "o",
    "DPT": "s",
    "SPOC": "^",
    "BC+PPO": "v",
    "VariBAD": "<",
    "AAWR": ">",
    "ADVISOR": "p",
    "BC": "h",
}

# Order for plotting (and legend)
METHOD_ORDER = ["Ours", "BC+PPO", "ADVISOR", "VariBAD", "AAWR", "BC", "SPOC", "DPT"]


# ============================================================================
# Data fetching utilities
# ============================================================================

def get_curve_from_wandb(run_url, metric):
    """Fetch a metric curve from a wandb run."""
    import wandb
    api = wandb.Api()
    run = api.run(run_url)
    # Use scan_history to avoid downsampling
    history = run.scan_history(keys=[metric])
    history = list(history)
    if len(history) == 0:
        return []
    history = {k: [d[k] for d in history] for k in history[0].keys()}
    curve = history[metric]
    return curve


def get_rl_curves(run_urls, metric):
    """Fetch learning curves from multiple wandb runs."""
    curves = []
    for run_url in run_urls:
        print(f"  Fetching {run_url}...")
        curve = get_curve_from_wandb(run_url, metric)
        if len(curve) > 0:
            curves.append(curve)
    return curves


def process_rl_data(run_urls, metric, steps_per_update, n_seeds):
    """
    Process RL wandb runs into (mean, std, steps) arrays.
    """
    curves = get_rl_curves(run_urls, metric)
    if len(curves) == 0:
        return None, None, None
    
    # Align to minimum length
    min_length = min(len(c) for c in curves)
    curves = [c[:min_length] for c in curves]
    curves = np.array(curves)
    
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    steps = np.arange(len(mean)) * steps_per_update
    
    return mean, std, steps


def process_ours_data(iterations_data):
    """
    Process our method's iteration data into arrays.
    iterations_data: list of (mean, std, steps_this_iteration)
    Returns: (means, stds, cumulative_steps)
    """
    means = np.array([d[0] for d in iterations_data])
    stds = np.array([d[1] for d in iterations_data])
    steps = np.array([d[2] for d in iterations_data])
    cumulative_steps = np.cumsum(steps)
    return means, stds, cumulative_steps


def resample_log_space(steps, values, n_points=LOG_RESAMPLE_POINTS):
    """
    Resample data to be evenly spaced in log space.
    This makes the curve look better on a log-scale x-axis.
    """
    # Filter out zero/negative steps
    valid_mask = steps > 0
    steps = steps[valid_mask]
    values = values[valid_mask]
    
    if len(steps) < 2:
        return steps, values
    
    # Create log-spaced x values
    log_steps_new = np.linspace(np.log10(steps[0]), np.log10(steps[-1]), n_points)
    steps_new = 10 ** log_steps_new
    
    # Interpolate values
    values_new = np.interp(steps_new, steps, values)
    
    return steps_new, values_new


def smooth_curve(values, window=RL_SMOOTH_WINDOW):
    """Apply sliding window smoothing."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='same')
    # Fix edge effects
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def get_log_spaced_marker_positions(xlim, n_markers=SUBSAMPLE_POINTS):
    """Get evenly log-spaced x positions for markers across xlim."""
    log_positions = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), n_markers)
    return 10 ** log_positions


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(output_path):
    """
    Create a 1 row x N column plot comparing all methods across tasks.
    Shared legend at the bottom.
    """
    # Publication-quality settings (matching plot_test_time_performance.py)
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
    
    n_tasks = len(TASK_NAMES)
    
    # Square-ish subplots
    subplot_size = 2.8
    fig_width = n_tasks * subplot_size + 1.5
    fig_height = subplot_size + 0.8
    
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height))
    
    # Handle single task case
    if n_tasks == 1:
        axes = [axes]
    
    for idx, task_name in enumerate(TASK_NAMES):
        ax = axes[idx]
        
        if task_name not in TASK_DATA:
            print(f"Warning: No data for task {task_name}")
            continue
        
        data = TASK_DATA[task_name]
        params = TASK_PARAMS[task_name]
        n_seeds = params["n_seeds"]
        xlim = params["xlim"]
        
        # Get marker positions (log-spaced across xlim)
        marker_x_positions = get_log_spaced_marker_positions(xlim, SUBSAMPLE_POINTS)
        
        # Plot expert as horizontal line (lower alpha)
        if "expert" in data:
            expert_mean, expert_std = data["expert"]
            ax.axhline(y=expert_mean, color=METHOD_COLORS["expert"], 
                       linestyle='--', linewidth=1.5, alpha=0.5, zorder=100)
        
        # Plot each method
        for method in METHOD_ORDER:
            if method not in data:
                continue
            
            method_data = data[method]
            color = METHOD_COLORS.get(method, "#333333")
            marker = METHOD_MARKERS.get(method, "o")
            
            if method == "Ours":
                # Our method: list of iteration tuples
                means, stds, steps = process_ours_data(method_data)
                stderr = stds / np.sqrt(n_seeds)
                
                # Plot line
                ax.plot(steps, means, '-', color=color, linewidth=2.5, zorder=20)
                
                # Plot error band
                ax.fill_between(steps, means - stderr, means + stderr,
                               alpha=0.15, color=color, linewidth=0, zorder=2)
                
                # Plot markers at each iteration point
                ax.scatter(steps, means, s=50, marker=marker, color=color,
                          edgecolors='white', linewidth=1.0, zorder=21)
            
            elif isinstance(method_data, list) and isinstance(method_data[0], str):
                # RL method: list of wandb URLs
                print(f"Loading {method} for {task_name} from wandb...")
                means, stds, steps = process_rl_data(
                    method_data, params["metric"], 
                    params["steps_per_update"], n_seeds
                )
                if means is None:
                    print(f"  Warning: No data for {method}")
                    continue
                
                stderr = stds / np.sqrt(n_seeds)
                
                # Resample to log space for even distribution on log-scale plot
                steps_log, means_log = resample_log_space(steps, means)
                _, stderr_log = resample_log_space(steps, stderr)
                
                # Smooth the resampled curves
                means_log = smooth_curve(means_log)
                stderr_log = smooth_curve(stderr_log)
                
                # Subsample for markers (in log space)
                if len(steps_log) > SUBSAMPLE_POINTS:
                    indices = np.linspace(0, len(steps_log) - 1, SUBSAMPLE_POINTS, dtype=int)
                    steps_sub = steps_log[indices]
                    means_sub = means_log[indices]
                else:
                    steps_sub = steps_log
                    means_sub = means_log
                
                # Plot line (log-resampled and smoothed)
                ax.plot(steps_log, means_log, '-', color=color, linewidth=2.0, zorder=10)
                
                # Plot error band
                ax.fill_between(steps_log, means_log - stderr_log, means_log + stderr_log,
                               alpha=0.15, color=color, linewidth=0, zorder=2)
                
                # Plot markers (subsampled in log space)
                ax.scatter(steps_sub, means_sub, s=40, marker=marker, color=color,
                          edgecolors='white', linewidth=1.0, zorder=11)
            
            else:
                # Offline method: single tuple (mean, std, steps)
                mean, std, steps_val = method_data
                stderr = std / np.sqrt(n_seeds)
                
                # Plot as dotted horizontal line spanning entire x range
                x_range = np.array(xlim)
                ax.plot(x_range, [mean, mean], ':', color=color, linewidth=2.0, zorder=10)
                
                # Error band across entire range
                ax.fill_between(x_range, mean - stderr, mean + stderr,
                               alpha=0.15, color=color, linewidth=0, zorder=2)
                
                # Plot markers evenly spaced in log space across xlim
                marker_y = np.full_like(marker_x_positions, mean)
                ax.scatter(marker_x_positions, marker_y, s=40, marker=marker, color=color,
                          edgecolors='white', linewidth=1.0, zorder=11)
        
        # Styling
        ax.set_title(task_name, pad=8)
        ax.set_xlabel('Environment Steps')
        if idx == 0:
            ax.set_ylabel(params["ylabel"])
        ax.set_xscale('log')
        
        # Set x limits
        current_xlim = list(xlim)
        if AUTO_XLIM_TO_DATA and "Ours" in data:
            _, _, ours_steps = process_ours_data(data["Ours"])
            first_nonzero = ours_steps[ours_steps > 0][0] if np.any(ours_steps > 0) else ours_steps[1]
            current_xlim[0] = first_nonzero * 0.9
        ax.set_xlim(current_xlim)
        ax.set_ylim(params["ylim"])
        
        # Set proper log-scale ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Grid for log scale
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#b0b0b0', 
                zorder=0, which='major')
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3, color='#b0b0b0', 
                zorder=0, which='minor')
        ax.set_axisbelow(True)
        
        # Full boundary/frame
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('black')
        
        # Set aspect ratio to make plots more square
        ax.set_box_aspect(1)
    
    # Create custom legend with line + marker
    legend_handles = []
    legend_labels = []
    
    # Use first task's data to determine what's in the legend
    first_task_data = TASK_DATA[TASK_NAMES[0]]
    
    # Expert
    if "expert" in first_task_data:
        legend_handles.append(Line2D([0], [0], color=METHOD_COLORS["expert"], 
                                     linestyle='--', linewidth=1.5, alpha=0.5))
        legend_labels.append('Expert')
    
    # Methods
    for method in METHOD_ORDER:
        if method not in first_task_data:
            continue
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        method_data = first_task_data[method]
        
        # Determine if offline method (dotted) or not (solid)
        is_offline = not (method == "Ours" or 
                         (isinstance(method_data, list) and isinstance(method_data[0], str)))
        linestyle = ':' if is_offline else '-'
        lw = 2.5 if method == "Ours" else 2.0
        
        handle = Line2D([0], [0], color=color, linestyle=linestyle, linewidth=lw,
                       marker=marker, markersize=7, markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.0)
        legend_handles.append(handle)
        legend_labels.append(method)
    
    # Shared legend at the bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', 
               ncol=len(legend_labels), bbox_to_anchor=(0.5, -0.02),
               frameon=True, fancybox=False, shadow=False,
               edgecolor='#cccccc', framealpha=1.0, facecolor='white',
               fontsize=12)
    
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
    
    print("Generating sample efficiency comparison plot...")
    plot_comparison(OUTPUT_DIR / "sample_efficiency_comparison.png")
    
    print("\nDone!")
