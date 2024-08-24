import numpy as np
import matplotlib.pyplot as plt
import os

def plot_path(coords, save_directory="plots", base_filename="path_plot"):
    """
    Plots the path of the user based on the provided coordinates and saves the plot as a PNG file.

    Args:
        coords (list or np.array): A list or array of coordinates where each coordinate is a tuple or array of (x, y, z) values.
        save_directory (str): The directory where the plot images will be saved. Defaults to "plots_2".
        base_filename (str): The base name for the saved plot files. The function appends an incremental number to this base name. Defaults to "path_plot".

    Returns:
        None
    """
    coords = np.array(coords)  # Convert the list of coordinates to a NumPy array.
    plt.figure()  # Create a new figure for the plot.
    plt.plot(coords[:, 0], coords[:, 2], 'ro-')  # Plot the X and Z coordinates, using red circles connected by lines.
    plt.xlabel('X Position (m)')  # Label for the X-axis.
    plt.ylabel('Z Position (m)')  # Label for the Z-axis.
    plt.title('Path of the user')  # Title of the plot.
    plt.xlim([-3, 3])  # Set the limits of the X-axis.
    plt.ylim([-3, 3])  # Set the limits of the Z-axis.
    plt.grid(True)  # Enable grid lines for better readability.

    save_plot_incrementally(save_directory, base_filename)  # Save the plot using the incremental naming function.
    plt.show()  # Display the plot on the screen.

def save_plot_incrementally(directory, base_filename):
    """
    Saves the plot with an incrementally numbered filename in the specified directory.

    Args:
        directory (str): The directory where the plot should be saved. If the directory does not exist, it will be created.
        base_filename (str): The base name for the plot file. A number is appended to this base name to create a unique filename.

    Returns:
        None
    """
    if not os.path.exists(directory):  # Check if the directory exists.
        os.makedirs(directory)  # If not, create the directory.

    # Get the list of files in the directory and extract numbers from filenames that match the pattern.
    files = os.listdir(directory)
    existing_numbers = [
        int(f.split('_')[-1].split('.')[0])
        for f in files
        if f.startswith(base_filename) and f.endswith('.png')
    ]
    # Determine the next number to use in the filename.
    next_number = max(existing_numbers) + 1 if existing_numbers else 1

    # Create the full filename with the incremented number.
    filename = f"{base_filename}_{next_number}.png"
    filepath = os.path.join(directory, filename)  # Join directory and filename to create the full file path.
    plt.savefig(filepath)  # Save the current plot to the specified file.
    print(f"Plot saved as {filepath}")  # Print a confirmation message with the file path.
