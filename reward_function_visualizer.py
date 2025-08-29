import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_functions_2d(std1, std2):
    """
    Generates and displays 2D plots for various kernel functions based on Euclidean
    distance from the origin.

    Args:
        std1 (float): The standard deviation for the 1-tanh function.
        std2 (float): The standard deviation for the exp function.
    """
    # --- 0. Confirm inputs ---
    print(f"Plotting with std1 (1-tanh) = {std1} and std2 (exp) = {std2}")

    # --- 1. Create the data grid ---
    # Define the range for x and y values
    x_range = np.linspace(-1, 1, 400)
    y_range = np.linspace(-1, 1, 400)
    # Create a meshgrid to represent every point in the 2D plane
    X, Y = np.meshgrid(x_range, y_range)

    # --- 2. Calculate the distance ---
    # Use the square of the Euclidean distance from the origin
    distance_sq = X**2 + Y**2

    # --- 3. Calculate the Z values for each function ---
    # These values will be represented by colors in the 2D plots.

    # Function 1: 1 - tanh(distance^2 / std1)
    if std1 == 0:
        # Handle the singularity; result is 0 for distance > 0, 1 otherwise
        Z1 = np.zeros_like(distance_sq)
        Z1[distance_sq == 0] = 1
    else:
        Z1 = 1 - np.tanh(distance_sq / std1)

    # Function 2: exp(-distance^2 / std2)
    if std2 == 0:
        # Handle the singularity; result is 0 for distance > 0, 1 otherwise
        Z2 = np.zeros_like(distance_sq)
        Z2[distance_sq == 0] = 1
    else:
        Z2 = np.exp(-distance_sq / std2)

    # Function 3: L2-Squared-Norm based kernel
    Z3 = 1 - distance_sq

    # Function 4: L2-Norm based kernel
    Z4 = 1 - np.sqrt(distance_sq)


    # --- 4. Create the 2D plots ---
    # Use plt.subplots for a cleaner layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Comparison of Kernel Functions (2D Top-Down View)', fontsize=16)

    # Define the extent for imshow to correctly label the axes
    extent = [x_range.min(), x_range.max(), y_range.min(), y_range.max()]

    # --- Plot for 1 - tanh(distance^2/std) ---
    ax1 = axes[0, 0]
    im1 = ax1.imshow(Z1, cmap='viridis', extent=extent, origin='lower')
    ax1.set_title(f'1 - tanh(distance² / {std1})', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # --- Plot for exp(-distance^2/std) ---
    ax2 = axes[0, 1]
    im2 = ax2.imshow(Z2, cmap='viridis', extent=extent, origin='lower')
    ax2.set_title(f'exp(-distance² / {std2})', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # Determine the common value range for the bottom two plots for consistent scaling
    vmin_magma = min(Z3.min(), Z4.min())
    vmax_magma = max(Z3.max(), Z4.max())

    # --- Plot for L2-Squared-Norm based kernel ---
    ax3 = axes[1, 0]
    im3 = ax3.imshow(Z3, cmap='magma', extent=extent, origin='lower', vmin=vmin_magma, vmax=vmax_magma)
    ax3.set_title('1 - (L2-Squared-Norm)', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    # --- Plot for L2-Norm based kernel ---
    ax4 = axes[1, 1]
    im4 = ax4.imshow(Z4, cmap='magma', extent=extent, origin='lower', vmin=vmin_magma, vmax=vmax_magma)
    ax4.set_title('1 - (L2-Norm)', fontsize=14)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    fig.colorbar(im4, ax=ax4, shrink=0.8)

    # --- 5. Display the plots ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.show()

if __name__ == '__main__':
    # --- Argument Parsing ---
    # Set up the parser to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot 2D representations of various kernel functions."
    )
    # Add arguments for the standard deviations
    parser.add_argument(
        'std1',
        type=float,
        help='Standard deviation for the 1-tanh function.'
    )
    parser.add_argument(
        'std2',
        type=float,
        help='Standard deviation for the exp function.'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main plotting function with the provided arguments
    plot_functions_2d(args.std1, args.std2)
