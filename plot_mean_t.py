import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_scalar_data(event_file, scalar_tag):
    """
    Extracts scalar data from a single tfevents file.
    """
    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if scalar_tag not in ea.Tags()['scalars']:
            return None
        scalar_events = ea.Scalars(scalar_tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        return pd.DataFrame({'step': steps, 'value': values})
    except Exception as e:
        print(f"Error processing file {os.path.basename(event_file)}: {e}")
        return None

# --- MODIFIED FUNCTION ---
def get_aggregated_data(root_dir, scalar_tag):
    """
    Finds all tfevents files, aggregates data, and calculates the mean and
    95% confidence interval.

    Returns:
        A tuple of (steps, mean_values, confidence_interval), or None if no data is found.
    """
    all_runs_data = []
    print(f"\n--- Processing Directory: {root_dir} for tag: {scalar_tag} ---")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if "tfevents" in file:
                event_file_path = os.path.join(subdir, file)
                run_data = extract_scalar_data(event_file_path, scalar_tag)
                if run_data is not None:
                    all_runs_data.append(run_data)

    if not all_runs_data:
        print(f"Warning: No data found for tag '{scalar_tag}' in {root_dir}. Skipping.")
        return None

    # Aggregate data from all runs
    aligned_df = pd.concat([df.set_index('step') for df in all_runs_data], axis=1)
    aligned_df = aligned_df.interpolate(method='linear', limit_direction='both', axis=0)
    aligned_df.fillna(method='ffill', inplace=True)
    aligned_df.fillna(method='bfill', inplace=True)

    # Calculate mean, standard deviation, and sample size
    mean_values = aligned_df.mean(axis=1)
    std_values = aligned_df.std(axis=1)
    steps = aligned_df.index
    # Get the number of runs (samples)
    n = aligned_df.shape[1]

    # Calculate 95% confidence interval
    # z-score for 95% confidence is 1.96
    confidence_interval = 1.96 * (std_values / np.sqrt(n))

    # Return confidence interval instead of standard deviation
    return steps, mean_values, confidence_interval

# --- MODIFIED FUNCTION ---
def plot_aggregated_data(ax, scalar_tag, path_suffixes, base_dir, colors):
    """
    Plots aggregated data with a 95% confidence interval on a given matplotlib axis.
    """
    for i, path_suffix in enumerate(path_suffixes):
        full_path = os.path.join(base_dir, path_suffix)
        # The third returned value is now the confidence interval
        aggregated_data = get_aggregated_data(full_path, scalar_tag)

        if aggregated_data:
            steps, mean_values, confidence_interval = aggregated_data

            color = colors[i % len(colors)]
            label = path_suffix.split('/')[-1]

            # Plot the mean line
            ax.plot(steps, mean_values, color=color['line'], linewidth=2, label=label)

            # Plot the shaded 95% confidence interval region
            ax.fill_between(
                steps,
                mean_values - confidence_interval,
                mean_values + confidence_interval,
                color=color['shade'],
                alpha=0.15
            )

            # Plot the faint boundary lines for the confidence interval
            ax.plot(steps, mean_values - confidence_interval, color=color['line'], linewidth=0.5, alpha=0.35)
            ax.plot(steps, mean_values + confidence_interval, color=color['line'], linewidth=0.5, alpha=0.35)

    # --- Final Touches on the specific subplot ---
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Aggregate and plot TensorBoard data from multiple experiment directories."
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help="The task in which the experiments lie."
    )
    parser.add_argument(
        '--path_suffixes',
        type=str,
        nargs='+',
        required=True,
        help="A list of experiment path suffixes to plot."
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        choices=['r', 'e'],
        required=True,
        default='r',
        help="Type of plot to generate: 'r' for reward, 'e' for end-effector errors."
    )
    args = parser.parse_args()

    # --- Configuration ---
    BASE_EXPERIMENTS_DIR = os.path.join('logs/rsl_rl', args.task)
    plt.style.use('seaborn-v0_8-whitegrid')
    color_palette = [
        {'line': 'darkgrey', 'shade': 'lightgrey'}, {'line': 'saddlebrown', 'shade': 'peru'},
        {'line': 'darkorange', 'shade': 'sandybrown'}, {'line': 'crimson', 'shade': 'lightcoral'},
        {'line': 'steelblue', 'shade': 'lightblue'}, {'line': 'seagreen', 'shade': 'mediumaquamarine'},
        {'line': 'teal', 'shade': 'paleturquoise'}, {'line': 'mediumpurple', 'shade': 'thistle'},
        {'line': 'darkslateblue', 'shade': 'lavender'}, {'line': 'olivedrab', 'shade': 'darkkhaki'},
    ]

    # --- Main Plotting Logic ---
    if args.plot_type == 'r':
        fig, ax = plt.subplots(figsize=(14, 8))
        scalar_tag_to_plot = 'Train/mean_reward'
        plot_aggregated_data(ax, scalar_tag_to_plot, args.path_suffixes, BASE_EXPERIMENTS_DIR, color_palette)
        ax.set_xlabel('Timesteps', fontsize=14)
        ax.set_ylabel('Mean Return', fontsize=14)

    elif args.plot_type == 'e':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), sharex=True)

        orientation_tag = 'Metrics/ee_pose/orientation_error'
        plot_aggregated_data(ax1, orientation_tag, args.path_suffixes, BASE_EXPERIMENTS_DIR, color_palette)
        ax1.set_ylabel('Orientation error (rad)', fontsize=12)

        position_tag = 'Metrics/ee_pose/position_error'
        plot_aggregated_data(ax2, position_tag, args.path_suffixes, BASE_EXPERIMENTS_DIR, color_palette)
        ax2.set_ylabel('Position Error (m)', fontsize=12)
        ax2.set_xlabel('Timesteps', fontsize=12)

    plt.tight_layout()
    plt.show()