#Use this file to plot NN results:
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import pandas as pd
from trophic_web_methods import *
from nn_converter import *
import os
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm




ORIGINAL_ARCTIC_VCOUNT = 145
ANTARCTICA_COLOR = '#545cd6'
ARCTIC_COLOR = "#35bcdb"
ANTARCTICA_DARK = "#2c35b5"
ARCTIC_DARK = "#1a7c93"
G_COLOR = "orange"



###################################################################################
###################################################################################
###################################################################################

tests = ["iris", "breast_cancer", "wine"]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), sharex=True)

for idx, test in enumerate(tests):
    results_dir = f"results/DAG/{test}/"

    num_trials = 5
    num_extinctions = 10

    arctic_results = np.zeros((num_extinctions, num_trials + 1))  # Add one for the percent-extinguished column
    g_arctic_results = np.zeros((num_extinctions, num_trials + 1))
    for i in range(num_extinctions):
        for j in range(1, num_trials + 1):  # Skip first column
            trial_path = f"DAG__{test}__{str(100 - i*10)}_perc_left__trial_{str(j-1)}.txt"
            current_path = os.path.join(results_dir, trial_path)
            current_dict = read_dict_from_txt(current_path)

            arctic_results[i, 0] = int(current_dict["arctic_ecount"])
            g_arctic_results[i, 0] = int(current_dict["g_ecount"])
            arctic_results[i, j] = float(current_dict["accuracy_arctic"])
            g_arctic_results[i, j] = float(current_dict["accuracy_g"])

    # Arctic results
    x = arctic_results[:, 0]  # Num edges Column
    y_mean_arctic = np.mean(arctic_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_arctic = np.std(arctic_results[:, 1:], axis=1)  # Standard deviation across trials
    axes[idx].errorbar(
        x, y_mean_arctic, yerr=y_std_arctic, fmt='o-', label=f"Arctic", capsize=5,
        ecolor=ARCTIC_DARK, alpha=0.5, color = ARCTIC_DARK
    )

    # Arctic Dense baseline results #Num edges Column
    x = g_arctic_results[:, 0]
    y_mean_g = np.mean(g_arctic_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_g = np.std(g_arctic_results[:, 1:], axis=1)  # Standard deviation across trials
    axes[idx].errorbar(
        x, y_mean_g, yerr=y_std_g, fmt='o-', label=f"Dense Baseline", capsize=5,
        ecolor=G_COLOR, alpha=0.5, color = G_COLOR
    )

    # Add labels, title, and legend for each subplot
    axes[idx].set_ylabel("Accuracy")
    axes[idx].set_title(f"Accuracy vs Number of Edges ({test})")
    #axes[idx].legend(framealpha = 1, loc='best')
    axes[idx].grid(True)

# Add a shared X-axis label
axes[-1].set_xlabel("Number of Hidden Edges left in each Network")


# Adjust layout to prevent overlap
plt.tight_layout()  # Leave space on the right for the legend

# Show the combined figure
plt.show()

###################################################################################
###################################################################################
###################################################################################

tests = ["mnist"]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharex=True)

for idx, test in enumerate(tests):
    results_dir = f"results/DAG/{test}/"

    num_trials = 5
    num_extinctions = 10

    arctic_results = np.zeros((num_extinctions, num_trials + 1))  # Add one for the percent-extinguished column
    g_arctic_results = np.zeros((num_extinctions, num_trials + 1))
    for i in range(num_extinctions):
        for j in range(1, num_trials + 1):  # Skip first column
            trial_path = f"DAG__{test}__{str(200 - i*20)}_perc_left__trial_{str(j-1)}.txt"
            current_path = os.path.join(results_dir, trial_path)
            current_dict = read_dict_from_txt(current_path)

            arctic_results[i, 0] = int(current_dict["arctic_ecount"])
            g_arctic_results[i, 0] = int(current_dict["g_ecount"])
            arctic_results[i, j] = float(current_dict["accuracy_arctic"])
            g_arctic_results[i, j] = float(current_dict["accuracy_g"])

    # Arctic results
    x = arctic_results[:, 0]  # Num edges Column
    y_mean_arctic = np.mean(arctic_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_arctic = np.std(arctic_results[:, 1:], axis=1)  # Standard deviation across trials
    axes.errorbar(
        x, y_mean_arctic, yerr=y_std_arctic, fmt='o-', label=f"Arctic", capsize=5,
        ecolor=ARCTIC_DARK, alpha=0.5, color = ARCTIC_DARK
    )

    # Arctic Dense baseline results #Num edges Column
    x = g_arctic_results[:, 0]
    y_mean_g = np.mean(g_arctic_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_g = np.std(g_arctic_results[:, 1:], axis=1)  # Standard deviation across trials
    axes.errorbar(
        x, y_mean_g, yerr=y_std_g, fmt='o-', label=f"Dense Baseline", capsize=5,
        ecolor=G_COLOR, alpha=0.5, color = G_COLOR
    )

    # Add labels, title, and legend for each subplot
    axes.set_ylabel("Accuracy")
    axes.set_title(f"Accuracy vs Number of Edges ({test})")
    #axes.legend(framealpha = 1, loc='best')
    axes.grid(True)

# Add a shared X-axis label
axes.set_xlabel("Number of Hidden Edges left in each Network")


plt.title("Arctic Hyper-compression testing with mnist")
# Adjust layout to prevent overlap
plt.tight_layout()  # Leave space on the right for the legend

# Show the combined figure
plt.show()


###################################################################################
###################################################################################
###################################################################################

tests = ["iris", "breast_cancer", "wine"]


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.5, 8), sharex=True)

for idx, test in enumerate(tests):
    results_dir = f"results/DAG/{test}/"

    num_trials = 5
    num_extinctions = 10
    
    results_dir = f"results/non-DAG/{test}/"
    antarctica_results = np.zeros((num_extinctions - 2, num_trials + 1))
    g_antarctica_results = np.zeros((num_extinctions - 2, num_trials + 1))

    for i in range(num_extinctions - 2):
        for j in range(1, num_trials + 1):  # Skip first column
            trial_path = f"non_DAG__{test}__{str(100 - i*10)}_perc_left__trial_{str(j-1)}.txt"
            current_path = os.path.join(results_dir, trial_path)
            current_dict = read_dict_from_txt(current_path)

            antarctica_results[i, 0] = int(current_dict["antarctica_ecount"])
            g_antarctica_results[i, 0] = int(current_dict["g_ecount"])
            antarctica_results[i, j] = float(current_dict["accuracy_antarctica"])
            g_antarctica_results[i, j] = float(current_dict["accuracy_g"])


        # Antarctica results
    x = antarctica_results[:, 0]  # Num edges Column
    y_mean_antarctica = np.mean(antarctica_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_antarctica = np.std(antarctica_results[:, 1:], axis=1)  # Standard deviation across trials
    axes[idx].errorbar(
        x, y_mean_antarctica, yerr=y_std_antarctica, fmt='o-', label=f"Antarctica", capsize=5,
        ecolor=ANTARCTICA_COLOR, alpha=0.5, color= ANTARCTICA_COLOR
    )


        # Antarcitc Dense baseline results #Num edges Column
    x = g_antarctica_results[:, 0]
    y_mean_g = np.mean(g_antarctica_results[:, 1:], axis=1)  # Mean accuracy across trials
    y_std_g = np.std(g_antarctica_results[:, 1:], axis=1)  # Standard deviation across trials
    axes[idx].errorbar(
        x, y_mean_g, yerr=y_std_g, fmt='o-', label=f"Dense Baseline", capsize=5,
        ecolor=G_COLOR, alpha=0.5, color = G_COLOR
    )

    # Add labels, title, and legend for each subplot
    axes[idx].set_ylabel("Accuracy")
    axes[idx].set_title(f"Accuracy vs Number of Edges ({test})")
    #axes[idx].legend(framealpha = 1, loc='best')
    axes[idx].grid(True)

# Add a shared X-axis label
axes[-1].set_xlabel("Number of Hidden Edges left in each Network")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined figure
plt.show()


###################################################################################
###################################################################################
###################################################################################

tests = ["iris", "breast_cancer", "wine"]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.5, 8))

data = []

for idx, test in enumerate(tests):
    results_dir = f"results/DAG/epoch_training/{test}/"

    num_trials = 3
    extinctions = list(range(10, 100, 10))
    epochs = list(range(100, 1100, 100))

    results_arctic = np.zeros((len(epochs), len(extinctions)))  
    arctic_x_labels = [0] * len(extinctions)
    arctic_y_labels = epochs

    results_g = np.zeros((len(epochs), len(extinctions)))  
    g_x_labels = [0] * len(extinctions)
    g_y_labels = epochs
    
    for i in extinctions:
        for k in epochs:
            current_arctic_values = []
            current_g_values = []
            
            for j in range(0, num_trials):  # Skip first column
                trial_path = f"DAG__{test}__{str(i)}_perc_left__epoch_{str(k)}__trial_{str(j)}.txt"
                current_path = os.path.join(results_dir, trial_path)
                current_dict = read_dict_from_txt(current_path)

                current_arctic_values.append(float(current_dict["accuracy_arctic"]))
                current_g_values.append(float(current_dict["accuracy_g"]))

                arctic_x_labels[extinctions.index(i)] = int(current_dict["arctic_ecount"])
                g_x_labels[extinctions.index(i)] = int(current_dict["g_ecount"])
            
            current_g_mean = max(current_g_values) 
            current_arctic_mean = max(current_arctic_values)

            results_arctic[epochs.index(k), extinctions.index(i)] = current_arctic_mean
            results_g[epochs.index(k), extinctions.index(i)] = current_g_mean


    # Sample input matrices (replace with your actual data)
    y_values = np.array(epochs)  # Replace with actual y-values if not 0-100
    x_g = np.array(g_x_labels)       # Replace with actual x-values for "g"
    x_arctic = np.array(arctic_x_labels)   # Replace with actual x-values for "arctic"

    # Flatten the matrices for interpolation
    points_g = np.array(np.meshgrid(x_g, y_values)).T.reshape(-1, 2)
    points_arctic = np.array(np.meshgrid(x_arctic, y_values)).T.reshape(-1, 2)

    values_g = results_g.flatten()
    values_arctic = results_arctic.flatten()

    # Create a common, linearly spaced grid for interpolation
    common_x = np.linspace(max(x_g.min(), x_arctic.min()), min(x_g.max(), x_arctic.max()), 50)
    common_y = y_values  # No need to interpolate y
    common_y = np.linspace(y_values.min(), y_values.max(), 50)

    X, Y = np.meshgrid(common_x, common_y)
    grid_points = np.c_[X.ravel(), Y.ravel()]

    # Interpolate
    g_interpolated = griddata(points_g, values_g, grid_points, method='linear').reshape(X.shape)
    arctic_interpolated = griddata(points_arctic, values_arctic, grid_points, method='linear').reshape(X.shape)
    
    

    interpolation_difference = arctic_interpolated - g_interpolated
    interpolation_maxes = np.maximum(arctic_interpolated, g_interpolated)

    data.append([interpolation_difference, interpolation_maxes])


    # Use the overall min and max to center at 0

    vmin = np.min(interpolation_difference)
    vmax = np.max(interpolation_difference)


    # Ensure vmin < 0 < vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


    current_contour = axes[idx].contourf(X, Y, interpolation_difference, levels=1000, cmap='bwr', norm=norm)
    fig.colorbar(current_contour, ax=axes[idx])

    axes[idx].set_title(f'Arctic Interpolation {test}')
    axes[idx].set_xlabel("Edges")
    axes[idx].set_ylabel("Epochs")





labels = []
# Click event handler
def on_click(event):
    if event.inaxes in axes:
        # Find the nearest data point
        x_idx = np.argmin(np.abs(X[0] - event.xdata))
        y_idx = np.argmin(np.abs(Y[:, 0] - event.ydata))
        
        # Get the value from the corresponding matrix

        value = data[np.where(axes == event.inaxes)[0][0]][1][y_idx, x_idx]
        
        # Add the label
        label = event.inaxes.text(event.xdata, event.ydata, f"{value:.2f}", 
                                  color="black", fontsize=8, ha='center', va='center', fontweight='bold',
                                  bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
        labels.append(label)
        plt.draw()

# Clear button callback
def clear_labels(event):
    for label in labels:
        label.remove()
    labels.clear()
    plt.draw()

# Add the event listeners
fig.canvas.mpl_connect("button_press_event", on_click)

# Add the clear button
clear_button_ax = fig.add_axes([0.01, 0.01, 0.025, 0.025])
clear_button = plt.Button(clear_button_ax, 'Clear Labels', color='gray', hovercolor='red')
clear_button.on_clicked(clear_labels)




# Adjust layout to prevent overlap
plt.tight_layout()  # Leave space on the right for the legend

# Show the combined figure
plt.show()

###################################################################################
###################################################################################
###################################################################################



###################################################################################
###################################################################################
###################################################################################

tests = ["iris", "breast_cancer", "wine"]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8))

data = []

for idx, test in enumerate(tests):
    results_dir = f"results/non-DAG/epoch_training/{test}/"

    num_trials = 3
    extinctions = list(range(20, 100, 10))
    epochs = list(range(100, 1100, 100))

    results_antarctica = np.zeros((len(epochs), len(extinctions)))  
    antarctica_x_labels = [0] * len(extinctions)
    antarctica_y_labels = epochs

    results_g = np.zeros((len(epochs), len(extinctions)))  
    g_x_labels = [0] * len(extinctions)
    g_y_labels = epochs
    
    for i in extinctions:
        for k in epochs:
            current_antarctica_values = []
            current_g_values = []
            
            for j in range(0, num_trials):  # Skip first column
                trial_path = f"non_DAG__{test}__{str(i)}_perc_left__epoch_{str(k)}__trial_{str(j)}.txt"
                current_path = os.path.join(results_dir, trial_path)
                current_dict = read_dict_from_txt(current_path)

                current_antarctica_values.append(float(current_dict["accuracy_antarctica"]))
                current_g_values.append(float(current_dict["accuracy_g"]))

                antarctica_x_labels[extinctions.index(i)] = int(current_dict["antarctica_ecount"])
                g_x_labels[extinctions.index(i)] = int(current_dict["g_ecount"])
            
            current_g_mean = max(current_g_values)
            current_antarctica_mean = max(current_antarctica_values)

            results_antarctica[epochs.index(k), extinctions.index(i)] = current_antarctica_mean
            results_g[epochs.index(k), extinctions.index(i)] = current_g_mean


    # Sample input matrices (replace with your actual data)
    y_values = np.array(epochs)  # Replace with actual y-values if not 0-100
    x_g = np.array(g_x_labels)       # Replace with actual x-values for "g"
    x_antarctica = np.array(antarctica_x_labels)   # Replace with actual x-values for "arctic"

    # Flatten the matrices for interpolation
    points_g = np.array(np.meshgrid(x_g, y_values)).T.reshape(-1, 2)
    points_antarctica = np.array(np.meshgrid(x_antarctica, y_values)).T.reshape(-1, 2)

    values_g = results_g.flatten()
    values_antarctica = results_antarctica.flatten()

    # Create a common, linearly spaced grid for interpolation
    common_x = np.linspace(max(x_g.min(), x_antarctica.min()), min(x_g.max(), x_antarctica.max()), 50)
    common_y = y_values  # No need to interpolate y
    common_y = np.linspace(y_values.min(), y_values.max(), 50)

    X, Y = np.meshgrid(common_x, common_y)
    grid_points = np.c_[X.ravel(), Y.ravel()]

    # Interpolate
    g_interpolated = griddata(points_g, values_g, grid_points, method='linear').reshape(X.shape)
    antarctica_interpolated = griddata(points_antarctica, values_antarctica, grid_points, method='linear').reshape(X.shape)
    
    

    interpolation_difference = antarctica_interpolated - g_interpolated
    interpolation_maxes = np.maximum(antarctica_interpolated, g_interpolated)

    data.append([interpolation_difference, interpolation_maxes])


    # Use the overall min and max to center at 0

    vmin = np.min(interpolation_difference)
    vmax = np.max(interpolation_difference)


    # Ensure vmin < 0 < vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


    current_contour = axes[idx].contourf(X, Y, interpolation_difference, levels=1000, cmap='bwr', norm=norm)
    fig.colorbar(current_contour, ax=axes[idx])

    axes[idx].set_title(f'Antarctic Interpolation {test}')
    axes[idx].set_xlabel("Edges")
    axes[idx].set_ylabel("Epochs")





labels = []
# Click event handler
def on_click(event):
    if event.inaxes in axes:
        # Find the nearest data point
        x_idx = np.argmin(np.abs(X[0] - event.xdata))
        y_idx = np.argmin(np.abs(Y[:, 0] - event.ydata))
        
        # Get the value from the corresponding matrix

        value = data[np.where(axes == event.inaxes)[0][0]][1][y_idx, x_idx]
        
        # Add the label
        label = event.inaxes.text(event.xdata, event.ydata, f"{value:.2f}", 
                                  color="black", fontsize=8, ha='center', va='center', fontweight='bold',
                                  bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
        labels.append(label)
        plt.draw()

# Clear button callback
def clear_labels(event):
    for label in labels:
        label.remove()
    labels.clear()
    plt.draw()

# Add the event listeners
fig.canvas.mpl_connect("button_press_event", on_click)

# Add the clear button
clear_button_ax = fig.add_axes([0.01, 0.01, 0.025, 0.025])
clear_button = plt.Button(clear_button_ax, 'Clear Labels', color='gray', hovercolor='red')
clear_button.on_clicked(clear_labels)




# Adjust layout to prevent overlap
plt.tight_layout()  # Leave space on the right for the legend

# Show the combined figure
plt.show()

###################################################################################
###################################################################################
###################################################################################