import numpy as np
import matplotlib.pyplot as plt

def plot_comparison():
    """
    Generates and displays a 2D plot comparing the functions
    1 - tanh(x) and exp(-x).
    """
    # --- 1. Create the data ---
    # Generate a range of x-values (representing distance) from 0 to 10
    # We use a fine grid for a smooth curve.
    x = np.linspace(0, 10, 400)

    # Calculate the corresponding y-values for each function
    y_tanh = 1 - np.tanh(x)
    y_exp = np.exp(-x)

    # --- 2. Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice-looking style for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both functions on the same axes for direct comparison
    ax.plot(x, y_tanh, label=r'$1 - \tanh(x)$', color='blue', linewidth=2.5)
    ax.plot(x, y_exp, label=r'$e^{-x}$', color='orangered', linewidth=2.5, linestyle='--')

    # --- 3. Add titles and labels for clarity ---
    ax.set_title('Comparison of Decay Functions', fontsize=16, fontweight='bold')
    ax.set_xlabel('Distance (x)', fontsize=12)
    ax.set_ylabel('Function Output', fontsize=12)

    # Set the limits for the axes to focus on the interesting part
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.1)

    # Add a legend to identify which line is which
    ax.legend(fontsize=12)

    # Add grid lines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 4. Display the plot ---
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Call the main plotting function
    plot_comparison()
