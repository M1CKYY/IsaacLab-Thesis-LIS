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


def get_aggregated_data(root_dir, scalar_tag):
    """
    Finds all tfevents files in a directory, aggregates the data for a
    specific scalar tag, and returns the processed data.

    Returns:
        A tuple of (steps, mean_values, std_values), or None if no data is found.
    """
    all_runs_data = []
    print(f"\n--- Processing Directory: {root_dir} ---")
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

    # Calculate mean and standard deviation
    mean_values = aligned_df.mean(axis=1)
    std_values = aligned_df.std(axis=1)
    steps = aligned_df.index
    return steps, mean_values, std_values


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Aggregate and plot TensorBoard data from multiple experiment directories."
    )
    # Use nargs='+' to accept one or more path suffixes
    parser.add_argument(
        '--path_suffixes',
        type=str,
        nargs='+',  # This allows for multiple arguments
        required=True,
        help="A list of experiment path suffixes to plot. The first is the baseline."
    )
    args = parser.parse_args()

    # --- Configuration ---
    BASE_EXPERIMENTS_DIR = 'logs/rsl_rl'
    SCALAR_TAG_TO_PLOT = 'Train/mean_reward'

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for the plots. The first is grey for the baseline.
    color_palette = [
        {'line': 'darkgrey', 'shade': 'darkgrey'},
        {'line': 'darkorange', 'shade': 'sandybrown'},
        {'line': 'tomato', 'shade': 'lightsalmon'},
        {'line': 'crimson', 'shade': 'lightcoral'},
        {'line': 'indianred', 'shade': 'lightpink'},
    ]

    # --- Main Loop for Plotting ---
    for i, path_suffix in enumerate(args.path_suffixes):
        full_path = os.path.join(BASE_EXPERIMENTS_DIR, path_suffix)

        aggregated_data = get_aggregated_data(full_path, SCALAR_TAG_TO_PLOT)

        if aggregated_data:
            steps, mean_values, std_values = aggregated_data

            # Get color for the current plot, looping if we run out
            color = color_palette[i % len(color_palette)]

            # Use the path suffix as the label for clarity
            label = path_suffix.split('/')[-1]  # Use the last part of the path as a label

            # Plot the mean line
            ax.plot(steps, mean_values, color=color['line'], linewidth=2.5, label=label)

            # Plot the shaded standard deviation region
            ax.fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color=color['shade'],
                alpha=0.15
            )

            ax.plot(steps, mean_values - std_values, color=color['line'], linewidth=0.5, alpha=0.35)
            ax.plot(steps, mean_values + std_values, color=color['line'], linewidth=0.5, alpha=0.35)

    # --- Final Touches on the Plot ---
    ax.set_xlabel('Timesteps', fontsize=14)
    #ax.set_ylabel(SCALAR_TAG_TO_PLOT, fontsize=14)
    ax.set_ylabel('Mean reward', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()