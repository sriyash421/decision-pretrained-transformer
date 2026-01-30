"""
Plot training-time performance comparison across methods.
X-axis: Environment steps
Y-axis: Total return (sum over test-time episodes)
For ICML paper - clean publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/checkpoint/siro/sriyash/dpt-code/results")
OUTPUT_DIR = Path("/checkpoint/siro/sriyash/dpt-code/plots")

TASKS = [
    "darkroom-easy",
    "darkroom-hard",
    "keydoor-markovian",
    "keydoor-nonmarkovian",
    "navigation-episodic",
    "navigation-nonepisodic",
]

SEEDS = ["seed0", "seed1", "seed2"]

# Horizon H for each task (used to calculate env steps for our method)
HORIZON = {
    "darkroom-easy": 100,
    "darkroom-hard": 200,
    "keydoor-markovian": 50,
    "keydoor-nonmarkovian": 50,
    "navigation-episodic": 20,
    "navigation-nonepisodic": 20,
}

# Number of iterations for each task
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
    "navigation-nonepisodic": "2D-Navigation-Nonepisodic",
}

# Expert performance for each task (total return = expert * n_test_episodes)
# Assuming 40 test episodes based on previous data
EXPERT_PERFORMANCE = {
    "darkroom-easy": 93,
    "darkroom-hard": 192,
    "keydoor-markovian": 95,
    "keydoor-nonmarkovian": 12,
    "navigation-episodic": 20,
    "navigation-nonepisodic": 20,
}

# Soft colors for methods
METHOD_COLORS = {
    "Ours": "#5B9BD5",             # Soft blue
    "DPT": "#70AD47",              # Soft green
    "SPOC": "#ED7D31",             # Soft orange
    "RL²": "#9E7CC3",              # Soft purple
    "VariBAD": "#A80000",          # Soft red
}

METHOD_ORDER = ["Ours", "DPT", "SPOC", "RL²", "VariBAD"]


# ============================================================================
# Data Loading
# ============================================================================

def calculate_cumulative_env_steps(task: str, iteration: int) -> int:
    """
    Calculate cumulative environment steps for our method at a given iteration.
    Data collected at iteration k: (k+1) * H * 10000
    Cumulative = sum_{i=0}^{k} (i+1) * H * 10000 = H * 10000 * (k+1)(k+2)/2
    """
    H = HORIZON[task]
    # Cumulative sum: sum_{i=0}^{k} (i+1) = (k+1)(k+2)/2
    cumulative = H * 1000 * (iteration + 1) * (iteration + 2) // 2
    return cumulative


def load_ours_training_curve(task: str, seed: str) -> dict:
    """Load training curve for our context accumulation method."""
    n_iters = TASK_ITERATIONS[task]
    env_steps = []
    total_returns = []
    
    for iteration in range(n_iters):
        # Path to eval_returns.npz
        subfolder = f"{task}-{seed}-{task}-{seed}"
        path = BASE_DIR / "context_accumulation_final_runs_lower_data" / task / seed / subfolder / f"dagger_step_{iteration}" / "eval" / "eval_returns.npz"
        
        if not path.exists():
            continue
        
        data = np.load(path)
        mean_returns = data["mean_returns"]
        
        # Total return = sum over all test-time episodes
        total_return = np.sum(mean_returns)
        
        # Cumulative env steps at this iteration
        steps = calculate_cumulative_env_steps(task, iteration)
        
        env_steps.append(steps)
        total_returns.append(total_return)
    
    if len(env_steps) == 0:
        return None
    
    return {
        "env_steps": np.array(env_steps),
        "total_returns": np.array(total_returns),
    }


def load_offline_method_final_return(method_folder: str, prefix: str, task: str, seed: str, path_suffix: str) -> float:
    """Load final total return for offline methods (DPT, SPOC)."""
    path = BASE_DIR / method_folder / f"{prefix}-{task}-{seed}" / path_suffix / "eval_returns.npz"
    
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    
    data = np.load(path)
    mean_returns = data["mean_returns"]
    return np.sum(mean_returns)


def load_online_method_training_curve(method_folder: str, prefix: str, task: str, seed: str) -> dict:
    """Load training curve for online methods (RL², VariBAD)."""
    path = BASE_DIR / method_folder / f"{prefix}-{task}-{seed}" / "eval_results.pkl"
    
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    
    with open(path, "rb") as f:
        eval_results = pickle.load(f)
    
    env_steps = []
    total_returns = []
    
    for entry in eval_results:
        env_steps.append(entry["timesteps"])
        total_returns.append(np.sum(entry["mean_returns"]))
    
    return {
        "env_steps": np.array(env_steps),
        "total_returns": np.array(total_returns),
    }


def aggregate_ours_across_seeds(task: str) -> dict:
    """Aggregate our method's training curves across seeds."""
    all_curves = []
    
    for seed in SEEDS:
        curve = load_ours_training_curve(task, seed)
        if curve is not None:
            all_curves.append(curve)
    
    if len(all_curves) == 0:
        return None
    
    # All seeds should have same env_steps
    env_steps = all_curves[0]["env_steps"]
    all_returns = np.array([c["total_returns"] for c in all_curves])
    
    mean_returns = np.mean(all_returns, axis=0)
    std_err = np.std(all_returns, axis=0, ddof=1) / np.sqrt(len(all_curves)) if len(all_curves) > 1 else np.zeros_like(mean_returns)
    
    return {
        "env_steps": env_steps,
        "mean": mean_returns,
        "std_err": std_err,
    }


def aggregate_offline_across_seeds(method_folder: str, prefix: str, task: str, path_suffix: str) -> float:
    """Aggregate offline method's final return across seeds."""
    returns = []
    
    for seed in SEEDS:
        ret = load_offline_method_final_return(method_folder, prefix, task, seed, path_suffix)
        if ret is not None:
            returns.append(ret)
    
    if len(returns) == 0:
        return None, None
    
    mean = np.mean(returns)
    std_err = np.std(returns, ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0
    
    return mean, std_err


def aggregate_online_across_seeds(method_folder: str, prefix: str, task: str) -> dict:
    """Aggregate online method's training curves across seeds."""
    all_curves = []
    
    for seed in SEEDS:
        curve = load_online_method_training_curve(method_folder, prefix, task, seed)
        if curve is not None:
            all_curves.append(curve)
    
    if len(all_curves) == 0:
        return None
    
    # All seeds should have same env_steps
    env_steps = all_curves[0]["env_steps"]
    all_returns = np.array([c["total_returns"] for c in all_curves])
    
    mean_returns = np.mean(all_returns, axis=0)
    std_err = np.std(all_returns, axis=0, ddof=1) / np.sqrt(len(all_curves)) if len(all_curves) > 1 else np.zeros_like(mean_returns)
    
    return {
        "env_steps": env_steps,
        "mean": mean_returns,
        "std_err": std_err,
    }


def build_results_dict() -> dict:
    """Build results dictionary for all methods and tasks."""
    results = {}
    
    for task in TASKS:
        results[task] = {}
        print(f"\n{task}:")
        
        # Ours (context accumulation)
        ours_data = aggregate_ours_across_seeds(task)
        results[task]["Ours"] = ours_data
        if ours_data is not None:
            print(f"  Ours: {len(ours_data['env_steps'])} points, "
                  f"final return = {ours_data['mean'][-1]:.2f}")
        
        # DPT (offline)
        dpt_mean, dpt_std = aggregate_offline_across_seeds(
            "dpt_final_runs_tuned_params", "dpt", task, "eval_epoch1000")
        results[task]["DPT"] = {"mean": dpt_mean, "std_err": dpt_std}
        if dpt_mean is not None:
            print(f"  DPT: final return = {dpt_mean:.2f}")
        
        # SPOC (offline)
        spoc_mean, spoc_std = aggregate_offline_across_seeds(
            "spoc_final_runs", "spoc", task, "eval")
        results[task]["SPOC"] = {"mean": spoc_mean, "std_err": spoc_std}
        if spoc_mean is not None:
            print(f"  SPOC: final return = {spoc_mean:.2f}")
        
        # RL² (online)
        rl2_data = aggregate_online_across_seeds(
            "rl2_debugged_final_runs", "rl2", task)
        results[task]["RL²"] = rl2_data
        if rl2_data is not None:
            print(f"  RL²: {len(rl2_data['env_steps'])} points, "
                  f"final return = {rl2_data['mean'][-1]:.2f}")
        
        # VariBAD (online)
        varibad_data = aggregate_online_across_seeds(
            "varibad_runs", "varibad", task)
        results[task]["VariBAD"] = varibad_data
        if varibad_data is not None:
            print(f"  VariBAD: {len(varibad_data['env_steps'])} points, "
                  f"final return = {varibad_data['mean'][-1]:.2f}")
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_training_curves(results: dict, output_path: Path):
    """
    Create a 1 row x 5 column plot showing training-time performance.
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
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',
    })
    
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(16, 3.5))
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_results = results[task]
        
        # Determine x-axis range from online methods
        max_steps = 0
        for method in ["Ours", "RL²", "VariBAD"]:
            data = task_results.get(method)
            if data is not None and "env_steps" in data:
                max_steps = max(max_steps, data["env_steps"][-1])
        
        if max_steps == 0:
            max_steps = 10_000_000  # Default
        
        # Plot expert as black dashed line
        expert_total = EXPERT_PERFORMANCE[task] * 40  # 40 test episodes
        ax.axhline(y=expert_total, color='black', linestyle='--', linewidth=2,
                   label='Expert', zorder=100)
        
        # Plot Ours (solid line with markers)
        ours_data = task_results.get("Ours")
        if ours_data is not None:
            color = METHOD_COLORS["Ours"]
            ax.fill_between(ours_data["env_steps"], 
                           ours_data["mean"] - ours_data["std_err"],
                           ours_data["mean"] + ours_data["std_err"],
                           alpha=0.2, color=color, linewidth=0)
            ax.plot(ours_data["env_steps"], ours_data["mean"], 'o-', 
                   color=color, linewidth=2, markersize=5,
                   label="Ours", zorder=20)
        
        # Plot DPT (dotted horizontal line)
        dpt_data = task_results.get("DPT")
        if dpt_data is not None and dpt_data["mean"] is not None:
            color = METHOD_COLORS["DPT"]
            ax.axhline(y=dpt_data["mean"], color=color, linestyle=':', 
                      linewidth=2.5, label="DPT", zorder=10)
            # Add std_err band
            if dpt_data["std_err"] is not None:
                ax.axhspan(dpt_data["mean"] - dpt_data["std_err"],
                          dpt_data["mean"] + dpt_data["std_err"],
                          alpha=0.15, color=color, linewidth=0)
        
        # Plot SPOC (dotted horizontal line)
        spoc_data = task_results.get("SPOC")
        if spoc_data is not None and spoc_data["mean"] is not None:
            color = METHOD_COLORS["SPOC"]
            ax.axhline(y=spoc_data["mean"], color=color, linestyle=':', 
                      linewidth=2.5, label="SPOC", zorder=10)
            if spoc_data["std_err"] is not None:
                ax.axhspan(spoc_data["mean"] - spoc_data["std_err"],
                          spoc_data["mean"] + spoc_data["std_err"],
                          alpha=0.15, color=color, linewidth=0)
        
        # Plot RL² (solid line)
        rl2_data = task_results.get("RL²")
        if rl2_data is not None:
            color = METHOD_COLORS["RL²"]
            ax.fill_between(rl2_data["env_steps"], 
                           rl2_data["mean"] - rl2_data["std_err"],
                           rl2_data["mean"] + rl2_data["std_err"],
                           alpha=0.2, color=color, linewidth=0)
            ax.plot(rl2_data["env_steps"], rl2_data["mean"], '-', 
                   color=color, linewidth=2, label="RL²", zorder=15)
        
        # Plot VariBAD (solid line)
        varibad_data = task_results.get("VariBAD")
        if varibad_data is not None:
            color = METHOD_COLORS["VariBAD"]
            ax.fill_between(varibad_data["env_steps"], 
                           varibad_data["mean"] - varibad_data["std_err"],
                           varibad_data["mean"] + varibad_data["std_err"],
                           alpha=0.2, color=color, linewidth=0)
            ax.plot(varibad_data["env_steps"], varibad_data["mean"], '-', 
                   color=color, linewidth=2, label="VariBAD", zorder=15)
        
        # Styling
        ax.set_title(TASK_NAMES[task], fontweight='bold', pad=10)
        ax.set_xlabel('Environment Steps')
        if idx == 0:
            ax.set_ylabel('Total Return')
        
        # Format x-axis with scientific notation
        ax.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
        
        # Enhanced grid
        ax.grid(True, which='major', alpha=0.4, linestyle='-', 
                linewidth=0.5, color='#cccccc', zorder=1)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', 
                linewidth=0.3, color='#dddddd', zorder=1)
        ax.minorticks_on()
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlim(0, max_steps * 1.05)
    
    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(METHOD_ORDER) + 1,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True,
               shadow=False, edgecolor='#cccccc')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved training curves plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    results = build_results_dict()
    
    # Save dictionary
    dict_path = OUTPUT_DIR / "training_time_results.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results dict to {dict_path}")
    
    # Create plot
    print("\nGenerating plot...")
    plot_training_curves(results, OUTPUT_DIR / "training_time_comparison.png")
    
    print("\nDone!")

