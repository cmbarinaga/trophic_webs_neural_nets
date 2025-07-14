from trophic_web_methods import *
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 

import torch
import torch.nn.functional as F

import mplcursors
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox


ANTARCTICA_COLOR = '#545cd6'
ARCTIC_COLOR = "#35bcdb"
ANTARCTICA_DARK = "#2c35b5"
ARCTIC_DARK = "#1a7c93"

random.seed(42)
np.random.seed(42)


#######################################################################
#######################################################################
#######################################################################
#######################################################################

#Formating Potter Data
file_path = "Potter_FW.txt"    # Replace with the actual path to your CSV file
csv_converted_path = "Potter_FW.csv"  # Replace with the desired path for the converted CSV file

with open(file_path, 'r') as file: #read file lines
    lines = file.readlines()

with open(csv_converted_path, 'w') as csv_file: #rewrite to CSV for easier parsing
    for line in lines:
        # Replace tabs with commas and write to the new CSV file
        temp = line.strip()
        csv_file.write(line.replace('\t', ','))


# Read the CSV file into a DataFrame
predation = pd.read_csv(csv_converted_path)


#Not sure if we want this line or not, bc duplicates may insinuate weight
predation = predation.drop_duplicates()

#set to get unique species names
names = set()

for row in predation.itertuples():
    names.update(row[1:])

names = list(names)


#Creating igraph Graph
edges = []

for row in predation.itertuples():
    edges.append( [names.index(row[2]), names.index(row[1])] )
    #Energy flows FROM PREY TO PREDATOR, predator is the first column, prey is the second column

antarctica = ig.Graph(edges = edges, directed = True)

antarctica.vs["name"] = names

#######################################################################
#Formating Norway Fjord Data


file_path = "Arctic_Food_Web_2016.xlsx"    # Path to xlsx

binary_predation_matrix = pd.read_excel(file_path)

csv_converted_path = "Arctic_Food_Web_2016.csv"  # Replace with the desired path for the converted CSV file

#Generating CSV to make it consistent with the other data and to get around issue of repeated labels on index axis
with open(csv_converted_path, 'w') as csv_file: 
    csv_file.write(f"Predator,Prey\n")
    for j in range(1, len(binary_predation_matrix.columns)):
        for i in range(0, len(binary_predation_matrix.columns) - 1):
            if binary_predation_matrix.iloc[i, j] == 1:
                csv_file.write(f"{binary_predation_matrix.columns[j]},{binary_predation_matrix.iloc[i, 0]}\n")


# Read the CSV file into a DataFrame
predation = pd.read_csv(csv_converted_path)


#Not sure if we want this line or not, bc duplicates may insinuate weight
#predation = predation.drop_duplicates()

names = set()

for row in predation.itertuples():
    names.update(row[1:])

names = list(names)

#Creating igraph Graph
edges = []

for row in predation.itertuples():
    edges.append( [names.index(row[2]), names.index(row[1])] )
    #Energy flows FROM PREY TO PREDATOR tonto

arctic = ig.Graph(edges = edges, directed = True)

arctic.vs["name"] = names


#######################################################################
#######################################################################
#######################################################################
#######################################################################

#Comparisons of the two networks:

comparison = {
    "Antarctica": {},
    "Arctic": {}
}

robustness_results = {
    "Antarctica": {},
    "Arctic": {}
}


#Static Network-wide comparisons

#1. Number of species
comparison["Arctic"]["Number of Species"] = arctic.vcount()
comparison["Antarctica"]["Number of Species"] = antarctica.vcount()

#2. Number of links
comparison["Arctic"]["Number of Links"] = arctic.ecount()
comparison["Antarctica"]["Number of Links"] = antarctica.ecount()

#3. Connectance (links / (species * (species - 1)))
comparison["Arctic"]["Connectance"] = arctic.ecount() / (arctic.vcount() * (arctic.vcount()))
comparison["Antarctica"]["Connectance"] = antarctica.ecount() / (antarctica.vcount() * (antarctica.vcount()))

#4. Average degree (links / species)
comparison["Arctic"]["Average Degree"] = arctic.ecount() / arctic.vcount()
comparison["Antarctica"]["Average Degree"] = antarctica.ecount() / antarctica.vcount()

#5. CDD
comparison["Arctic"]["CDD"] = cumulative_degree_distribution(arctic, plot = False)
comparison["Antarctica"]["CDD"] = cumulative_degree_distribution(antarctica, plot = False)

#6. CDD Fitting
comparison["Arctic"]["CDD Fit"] = fit_CDD(arctic, plot = False)[0] #Fetch name
comparison["Antarctica"]["CDD Fit"] = fit_CDD(antarctica, plot = False)[0]

#7. Modularity
comparison["Arctic"]["Modularity"] = role_identifier(arctic).modularity
comparison["Antarctica"]["Modularity"] = role_identifier(antarctica).modularity

#Static Nodal Comparisons

#8. KSI (avg and std for each graph)
ksi_indexer(arctic)
ksi_indexer(antarctica)

comparison["Arctic"]["KSI"] = [f'Mean: {np.mean(np.array(arctic.vs["ksi"]))}', f'STD: {np.std(np.array(arctic.vs["ksi"]))}']
comparison["Antarctica"]["KSI"] = [f'Mean: {np.mean(np.array(antarctica.vs["ksi"]))}', f'STD: {np.std(np.array(antarctica.vs["ksi"]))}']


#9. Trophic Level (avg and std for each graph)
trophic_level_indexer(arctic)
trophic_level_indexer(antarctica)

comparison["Arctic"]["Trophic Level"] = [f'Mean: {np.mean(np.array(arctic.vs["trophic_level"]))}', f'STD: {np.std(np.array(arctic.vs["trophic_level"]))}']
comparison["Antarctica"]["Trophic Level"] = [f'Mean: {np.mean(np.array(antarctica.vs["trophic_level"]))}', f'STD: {np.std(np.array(antarctica.vs["trophic_level"]))}']

#10. Role (percent of each role present)
comparison["Arctic"]["Role"] = []
comparison["Antarctica"]["Role"] = []

roles = ["Network Connector", "Module Connector", "Module Specialist", "Module Hub"]

for j in roles:
    comparison["Arctic"]["Role"].append(len(arctic.vs.select(role = j)))
    comparison["Antarctica"]["Role"].append(len(antarctica.vs.select(role = j)))

for j in range(len(roles)):
    comparison["Arctic"]["Role"][j] = f"{roles[j]}: {comparison['Arctic']['Role'][j] / arctic.vcount() * 100}%"
    comparison["Antarctica"]["Role"][j] = f"{roles[j]}: {comparison['Antarctica']['Role'][j] / antarctica.vcount() * 100}%"


#11. Notable species
#Fetch top 10 KSI species in each
comparison["Arctic"]["Notable Species"] = {}

for i in range(0, 10):
    temp = arctic.vs["ksi"].index(sorted(arctic.vs["ksi"], reverse = True)[i])

    comparison["Arctic"]["Notable Species"][arctic.vs[temp]["name"]] = {
        "KSI": arctic.vs[temp]["ksi"],
        "Trophic Level": arctic.vs[temp]["trophic_level"],
        "Role": arctic.vs[temp]["role"]
    }

comparison["Antarctica"]["Notable Species"] = {}

for i in range(0, 10):
    temp = antarctica.vs["ksi"].index(sorted(antarctica.vs["ksi"], reverse = True)[i])

    comparison["Antarctica"]["Notable Species"][antarctica.vs[temp]["name"]] = {
        "KSI": antarctica.vs[temp]["ksi"],
        "Trophic Level": antarctica.vs[temp]["trophic_level"],
        "Role": antarctica.vs[temp]["role"]
    }

#11.25: Number of basal and apex species

comparison["Arctic"]["Basal Species"] = sum(1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") == 0 and arctic.degree(i, mode = "out") > 0)
comparison["Arctic"]["Apex Species"] = sum(1 for i in range(arctic.vcount()) if arctic.degree(i, mode = "in") > 0 and arctic.degree(i, mode = "out") == 0)

print("Arctic Basal Species: " + str(comparison["Arctic"]["Basal Species"]) + " Arctic Apex Species: " + str(comparison["Arctic"]["Apex Species"]) )

has_cycles = not arctic.is_dag()
print("Cycles exist in the arctic graph." if has_cycles else "No cycles in the arctic graph.")

comparison["Antarctica"]["Basal Species"] = sum(1 for i in range(antarctica.vcount()) if antarctica.degree(i, mode = "in") == 0 and antarctica.degree(i, mode = "out") > 0)
comparison["Antarctica"]["Apex Species"] = sum(1 for i in range(antarctica.vcount()) if antarctica.degree(i, mode = "in") > 0 and antarctica.degree(i, mode = "out") == 0)

print("Antarctica Basal Species: " + str(comparison["Antarctica"]["Basal Species"]) + " Antarctica Apex Species: " + str(comparison["Antarctica"]["Apex Species"]) )

has_cycles = not antarctica.is_dag()
print("Cycles exist in the antarctica graph." if has_cycles else "No cycles in the antarctica graph.")


apex_leveling(antarctica, "antarctica", min_or_max="max")
apex_leveling(arctic, "arctic", min_or_max="min")



#11.5 Plotting histograms of KSI and Trophic Levels:
# 1. Histogram for KSI
fig_ksi, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 2 rows, 1 column

# Precompute bins for KSI
bins_ksi = np.histogram_bin_edges(
    np.concatenate([antarctica.vs["ksi"], arctic.vs["ksi"]]), bins=20
)

# Antarctica KSI Histogram
antarctica_ksi_counts, _ = np.histogram(antarctica.vs["ksi"], bins=bins_ksi, density=True)
arctic_ksi_counts, _ = np.histogram(arctic.vs["ksi"], bins=bins_ksi, density=True)

# Determine the maximum y-value for consistent y-ticks
max_ksi_density = max(max(antarctica_ksi_counts), max(arctic_ksi_counts))

# Antarctica KSI Histogram
ax1.hist(antarctica.vs["ksi"], bins=bins_ksi, color=ANTARCTICA_COLOR, alpha=0.7, edgecolor="black", density=True)
ax1.set_title("Antarctica: KSI Distribution")
ax1.set_xlabel("KSI")
ax1.set_ylabel("Density")
ax1.set_xticks(bins_ksi)  # Add ticks at all bin edges
ax1.set_yticks(np.linspace(0, max_ksi_density, 5))  # Set consistent y-ticks
ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Arctic KSI Histogram
ax2.hist(arctic.vs["ksi"], bins=bins_ksi, color=ARCTIC_COLOR, alpha=0.7, edgecolor="black", density=True)
ax2.set_title("Arctic: KSI Distribution")
ax2.set_xlabel("KSI")
ax2.set_ylabel("Density")
ax2.set_xticks(bins_ksi)  # Add ticks at all bin edges
ax2.set_yticks(np.linspace(0, max_ksi_density, 5))  # Set consistent y-ticks
ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the KSI histogram figure
plt.show()

# 2. Histogram for Trophic Levels
fig_trophic, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))  # 2 rows, 1 column

# Precompute bins for Trophic Levels
bins_trophic = np.histogram_bin_edges(
    np.concatenate([antarctica.vs["trophic_level"], arctic.vs["trophic_level"]]), bins=20
)

# Antarctica Trophic Level Histogram
antarctica_trophic_counts, _ = np.histogram(antarctica.vs["trophic_level"], bins=bins_trophic, density=True)
arctic_trophic_counts, _ = np.histogram(arctic.vs["trophic_level"], bins=bins_trophic, density=True)

# Determine the maximum y-value for consistent y-ticks
max_trophic_density = max(max(antarctica_trophic_counts), max(arctic_trophic_counts))

# Antarctica Trophic Level Histogram
ax3.hist(antarctica.vs["trophic_level"], bins=bins_trophic, color=ANTARCTICA_COLOR, alpha=0.7, edgecolor="black", density=True)
ax3.set_title("Antarctica: Trophic Level Distribution")
ax3.set_xlabel("Trophic Level")
ax3.set_ylabel("Density")
ax3.set_xticks(bins_trophic)  # Add ticks at all bin edges
ax3.set_yticks(np.linspace(0, max_trophic_density, 5))  # Set consistent y-ticks
ax3.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Arctic Trophic Level Histogram
ax4.hist(arctic.vs["trophic_level"], bins=bins_trophic, color=ARCTIC_COLOR, alpha=0.7, edgecolor="black", density=True)
ax4.set_title("Arctic: Trophic Level Distribution")
ax4.set_xlabel("Trophic Level")
ax4.set_ylabel("Density")
ax4.set_xticks(bins_trophic)  # Add ticks at all bin edges
ax4.set_yticks(np.linspace(0, max_trophic_density, 5))  # Set consistent y-ticks
ax4.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the Trophic Level histogram figure
plt.show()


#11.75: Writing the comparison to readable text file
# Export the comparison dictionary to a text file
with open("comparison.txt", "w") as text_file:
    for category, metrics in comparison.items():
        text_file.write(f"{category}:\n")
        for metric, value in metrics.items():
            if isinstance(value, list):
                text_file.write(f"  {metric}: {', '.join(value)}\n")
            else:
                text_file.write(f"  {metric}: {value}\n")
        text_file.write("\n")

print("Comparison dictionary exported to comparison.txt")

print("Arctic species that is a Network connector: " + str(arctic.vs.select(role = "Network Connector")["name"] ))
print("Antarctic species that is a Network connector: " + str(antarctica.vs.select(role = "Network Connector")["name"]))


compare_and_color_modules(antarctica, "antarctica")
compare_and_color_modules(arctic, "arctic")






#12. Dynamic Iterative Extinctions:
# KSI/Trophic level recomputed each iteration, taking into account new extinctions
#In each newly evaluated iteration, the species that fits the criteria top/bottom/median for ksi/trophic is extinguished
figure_count = 1
thresholds = [0.25, 0.5, 0.75]
metrics = ["ksi", "trophic_level"]

for threshold in thresholds:
    for metric in metrics:
        if metric == "ksi":
            types = ["top", "bottom"]
        else:
            types = ["top", "median"]

        for type_ in types:
    
            #Antarctica
            antarctica_collapse = 0 #initializing collapse point
            temp_g = antarctica.copy()
            antarctica_x = list(range(antarctica.vcount()))
            antarctica_y = []
            max_delta = 0 #to track the max delta between points
    
            #All species KSI for each extinction
            by_species_antarctica = {}
            for name in antarctica.vs["name"]:
                by_species_antarctica[name] = []
    
            #Ordered list of what animals are being extinguished under each iteration
            antarctica_pop_list = []
    
            for i in antarctica_x:
                #Updating KSI values
                trophic_level_indexer(temp_g)
                ksi_indexer(temp_g)
                for j in range(temp_g.vcount()):
                    by_species_antarctica[temp_g.vs[j]["name"]].append(temp_g.vs[j]["ksi"])
    
    
                #Checking if collapsed
                components = temp_g.components(mode="weak")
                if len(components) > 1 and antarctica_collapse == 0 and all(x > 1 for x in sorted(components.sizes(), reverse=True)[0:2]):
                    antarctica_collapse = i
                
                #Adding the current population to the y axis
                antarctica_y.append(temp_g.vcount())
    
                #Tracking largest delta between points
                if len(antarctica_y) > 1:
                    antarctica_pop_list.append(popped)
                    if antarctica_y[-2] - antarctica_y[-1] > max_delta:
                        antarctica_mark = (popped, i - 1) #does this record the right point idk
                        max_delta = antarctica_y[-2] - antarctica_y[-1]
    
                #Dynamic extinction simulation
                temp_g, popped = extinction_simulation(temp_g, type_ = type_, metric=metric, percent=0, threshold=threshold)
                
            antarctica_pop_list.append("END")
            antarctica_counts = antarctica_y.copy()
            antarctica_r50 = robustness_50(antarctica_counts, antarctica.vcount())
            robustness_results["Antarctica"][f"{metric}_{type_}_thr{int(threshold*100)}"] = antarctica_r50
            print(f"Antarctica R50 ({metric} {type_}, threshold {threshold}): {antarctica_r50}")
            antarctica_x = [x / antarctica.vcount() * 100 for x in antarctica_x]
            antarctica_y = [y / antarctica.vcount() * 100 for y in antarctica_counts]
    
    
    
            #Arctic
            max_delta = 0 
            arctic_collapse = 0
            temp_g = arctic.copy()
            arctic_x = list(range(arctic.vcount()))
            arctic_y = []
            
            by_species_arctic = {}
            for name in arctic.vs["name"]:
                by_species_arctic[name] = []
    
            arctic_pop_list = []
    
            for i in arctic_x:
    
                ksi_indexer(temp_g)
                trophic_level_indexer(temp_g)
                for j in range(temp_g.vcount()):
                    by_species_arctic[temp_g.vs[j]["name"]].append(temp_g.vs[j]["ksi"])
    
    
                components = temp_g.components(mode="weak")
                if len(components) > 1 and arctic_collapse == 0 and all(x > 1 for x in sorted(components.sizes(), reverse=True)[0:2]):
                    arctic_collapse = i
                arctic_y.append(temp_g.vcount())
    
                if len(arctic_y) > 1:
                    arctic_pop_list.append(popped)
                    if arctic_y[-2] - arctic_y[-1] > max_delta:
                        max_delta = arctic_y[-2] - arctic_y[-1]
                        arctic_mark = (popped, i - 1) #does this record the right point idk
                        
    
                temp_g, popped = extinction_simulation(temp_g, type_ = type_, metric=metric, percent=0, threshold = threshold)
    
            arctic_counts = arctic_y.copy()
            arctic_r50 = robustness_50(arctic_counts, arctic.vcount())
            robustness_results["Arctic"][f"{metric}_{type_}_thr{int(threshold*100)}"] = arctic_r50
            print(f"Arctic R50 ({metric} {type_}, threshold {threshold}): {arctic_r50}")
            arctic_x = [x / arctic.vcount() * 100 for x in arctic_x]
            arctic_y = [y / arctic.vcount() * 100 for y in arctic_counts]

            arctic_pop_list.append("END")
    
    
            #Plotting the results
            fig, ax1 = plt.subplots()
            plt.subplots_adjust(bottom=0.25, right = 0.725)
    
            pop_plot_objects = []
    
            #Baseline y = 100 - x reference line
            pop_plot_objects.append(ax1.plot([0, max(antarctica_x)], [max(antarctica_y), 0], label="Reference", color="gray", linewidth = 1, linestyle = "--", alpha = 0.5)[0])  # Line plot for y1
            
            #Points at which max population delta occurs
            pop_plot_objects.append(ax1.scatter(arctic_x[arctic_mark[1]], arctic_y[arctic_mark[1]], label = arctic_mark[0], color = ARCTIC_COLOR, s = 100))
            pop_plot_objects.append(ax1.scatter(antarctica_x[antarctica_mark[1]], antarctica_y[antarctica_mark[1]], label = antarctica_mark[0], color = ANTARCTICA_COLOR, s = 100))
    
            #Plotting the extinction lines
            pop_plot_objects.append(ax1.plot(antarctica_x, antarctica_y, label="Antarctic Extinction", color=ANTARCTICA_COLOR, linewidth = 4)[0])  # Line plot for y1
            if antarctica_collapse != 0:
                pop_plot_objects.append(ax1.axhline(antarctica_y[antarctica_collapse], color=ANTARCTICA_COLOR, linestyle='dotted', linewidth=2, label="Antarctic Collapse"))
    
            pop_plot_objects.append(ax1.plot(arctic_x, arctic_y, label="Arctic Extinction", color=ARCTIC_COLOR, linewidth = 4)[0])  # Line plot for y2
            if arctic_collapse != 0:
                pop_plot_objects.append(ax1.axhline(arctic_y[arctic_collapse], color=ARCTIC_COLOR, linestyle='dotted', linewidth=2, label="Arctic Collapse"))
            #Darker Colors:
            #Arctic: #0f8aa6
            #Antarctica: #020ca1
    
            #Controlling transparency of plots of population
    
            # Add a slider for transparency
            ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])  # Position [left, bottom, width, height]
            slider = Slider(ax_slider, "Alpha", 0, 1.0, valinit=0.5)  # Slider range from 0.1 to 1.0
    
            # Update function for the slider
            def update_alpha(val):
                for line in pop_plot_objects:
                    line.set_alpha(slider.val)  # Update the alpha value of the line
                fig.canvas.draw_idle()  # Redraw the figure
    
            # Connect the slider to the update function
            slider.on_changed(update_alpha)
    
    
            # Add labels, title, and legend
            ax1.set_xlabel("Percent of ORIGINAL population Extinguished (Dynamically deleting the newest extrema)")
            ax1.set_ylabel("Percent of Population Left")
    
            ax2 = ax1.twinx()
    
            label_ax = plt.axes([0.85, 0.6, 0.10, 0.05])  # [left, bottom, width, height]
            coord_ax = plt.axes([0.85, 0.5, 0.10, 0.05])
    
    
            label_box = TextBox(label_ax, 'Highlight Line:', initial='')
            coord_box = TextBox(coord_ax, 'Probe X:', initial='')
    
    
            lines = []
    
            for key in by_species_antarctica:
                line, = ax2.plot(antarctica_x[:len(by_species_antarctica[key])], by_species_antarctica[key], color = ANTARCTICA_COLOR, label = key, alpha = 0.3, picker = 2)
                lines.append(line)
    
    
            for key in by_species_arctic:
                line, = ax2.plot(arctic_x[:len(by_species_arctic[key])], by_species_arctic[key], color = ARCTIC_COLOR, label = key, alpha = 0.3, picker = 2)
                lines.append(line)
             
            #Straight from ChatGPT:
            # Store original line widths and states
            original_lws = {line: line.get_linewidth() for line in lines}
            original_colors = {line: line.get_color() for line in lines}
            active_lines = {line: False for line in lines}  # Track toggled state
    
            # Annotations per line
            annotations = {}
            probe_annots = []
        
            def toggle_line_by_click(event):
                line = event.artist
                if line in lines:
                    x = event.mouseevent.xdata
                    y = event.mouseevent.ydata
                    toggle_line(line, x, y)
    
            def toggle_line(line, x=None, y=None):
                active_lines[line] = not active_lines[line]
    
                if active_lines[line]:
                    line.set_linewidth(2)
                    line.set_alpha(1)
                    if original_colors[line] == ANTARCTICA_COLOR:
                        line.set_color(ANTARCTICA_DARK)  # for antarctica
                    else:
                        line.set_color(ARCTIC_DARK)  # for arctic
                    line.set_zorder(10)
    
                    # Determine new arrow base point
                    anchor_point = (x, y) if x is not None and y is not None else (line.get_xdata()[0], line.get_ydata()[0])
    
                    # Remove previous annotation if it exists (to update xy)
                    if line in annotations:
                        annotations[line].remove()
    
                    annot = ax2.annotate(
                        line.get_label(),
                        xy=anchor_point,
                        xytext=(30, 30),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0, relpos=(0, 0)),
                        fontsize=10,
                    )
                    annotations[line] = annot
                    annotations[line].set_picker(True)
                    annotations[line].draggable(True)
    
                else:
                    line.set_linewidth(original_lws[line])
                    line.set_alpha(0.3)
                    line.set_color(original_colors[line])
                    line.set_zorder(1)
                    if line in annotations:
                        annotations[line].set_visible(False)
    
                fig.canvas.draw_idle()
            
    
            def submit_text(label_input):
                label_input = label_input.strip()
                for line in lines:
                    if line.get_label().lower() == label_input.lower():
                        toggle_line(line)
                        break
    
            vertical_line = None
    
            def submit_probe(x_input):
                global vertical_line
    
                
                try:
                    x_val = float(x_input.strip())
                except ValueError:
                    print("Invalid x value")
                    return
    
                # Clear previous annotations
                for annot in probe_annots:
                    annot.remove()
                probe_annots.clear()
    
                if vertical_line:
                    vertical_line.remove()
                
                closest_index = []
                closest_index.append(min(range(len(arctic_x)), key= lambda i: abs(arctic_x[i] - x_val)))
                closest_index.append(min(range(len(antarctica_x)), key= lambda i: abs(antarctica_x[i] - x_val)))
                
                to_post = (None, None, None)
    
                if abs(arctic_x[closest_index[0]] - x_val) < abs(antarctica_x[closest_index[1]] - x_val):
                    if arctic_pop_list[closest_index[0]]:
                        to_post = ("Arctic", arctic_pop_list[closest_index[0]][0], arctic_x[closest_index[0]] )
                    else:
                        to_post = ("Arctic", arctic_pop_list[closest_index[0]], arctic_x[closest_index[0]] )
    
                else:
                    if antarctica_pop_list[closest_index[1]]:
                        to_post = ("Antarctica", antarctica_pop_list[closest_index[1]][0], antarctica_x[closest_index[1]])
                    else:
                        to_post = ("Antarctica", antarctica_pop_list[closest_index[1]], antarctica_x[closest_index[1]])
                # Loop over each line to find closest x point
                
                if to_post[0] == "Arctic":
                    vertical_line = ax2.axvline(x=to_post[2], color=ARCTIC_COLOR, linestyle=':', alpha = 1, linewidth = 0.75)
                else:
                    vertical_line = ax2.axvline(x=to_post[2], color=ANTARCTICA_COLOR, linestyle=':', alpha = 1, linewidth = 0.75)
    
                y_top = ax2.get_ylim()[1]
                spacing = (y_top - ax2.get_ylim()[0]) * 0.05
                
                y_pos = y_top - spacing
                annot = ax2.annotate(
                    f"{to_post[0]}: {to_post[1]}",
                    xy=(to_post[2], y_pos),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round", fc="white"),
                    ha='left'
                )
                probe_annots.append(annot)
    
                fig.canvas.draw_idle()
    
            fig.canvas.mpl_connect("pick_event", toggle_line_by_click)
            label_box.on_submit(submit_text)
            coord_box.on_submit(submit_probe)
    
    
            ax2.set_ylabel("KSI of remaining species")
    
    
            ax1.set_title(f"Dynamic Extinction vs Resultant Population using {metric} from {type_}", loc = "center")
            
            handles, labels = ax1.get_legend_handles_labels()
            
            ax1.legend(handles, labels, loc = "upper right")
            plt.grid(True)
    
            # Show the plot
            plt.show()
            figure_count += 1

print("Robustness-50 Results:")
for web, results in robustness_results.items():
    for key, value in results.items():
        print(f"{web} - {key}: {value}")













        


#14. Plot within degress and connectivity on graph

#For Antarctica
plt.scatter(np.array(antarctica.vs["among_module_connectivity"]), np.array(antarctica.vs["within_module_degree"]), s=(np.array(antarctica.vs["trophic_level"])**4), c=antarctica.vs["module"]  )    
plt.axhline(y=2.5, color='red', linestyle='--', linewidth=1, label="Horizontal Line")
plt.axvline(x=0.62, color='blue', linestyle='--', linewidth=1, label="Vertical Line")
plt.ylabel("Within-Module Degree (Z)")
plt.xlabel("Among-Module Connectivity (C)")
plt.title("Antarctic Within-Module Degree vs. Among-Module Connectivity")
plt.show()

#For Arctic
plt.scatter(np.array(arctic.vs["among_module_connectivity"]), np.array(arctic.vs["within_module_degree"]), s=(np.array(arctic.vs["trophic_level"])**4), c=arctic.vs["module"]  )    
plt.axhline(y=2.5, color='red', linestyle='--', linewidth=1, label="Horizontal Line")
plt.axvline(x=0.62, color='blue', linestyle='--', linewidth=1, label="Vertical Line")
plt.ylabel("Within-Module Degree (Z)")
plt.xlabel("Among-Module Connectivity (C)")
plt.title("Arctic Within-Module Degree vs. Among-Module Connectivity")
plt.show()



"""
Extinctions by...
- trophic level (top/down, bottom/up, wasp waist)
- KSI

Nodal comparisons:
- impact of extinction
- KSI
- identify modules (sub functions) of each net, and identify species that connect modules 
    these species keep net cohesion
    amount of modules doesn't matter as much as what species connect the two
    How to plot: https://www.researchgate.net/figure/Plot-of-within-module-degree-Z-and-among-module-connectivity-C-The-threshold-values_fig5_333983593
    Setting limits: https://www.nature.com/articles/nature03288

    
For NN translation:
- translate nodes into neurons via trophic level / max trophic level per module
- compare topological roles to weight values arrived at in training
- split up neuron pathways by module, i.e. have as many ins/outs as modules to normalize module size better
    - i.e. think of ways to translate/reduce outside of just primary producers to apex
        run into issue of needing to run extinctions on an artificial web, but look into it

        
In comparison:
- Look at the basics to characterize complexity of the web (number of connections total, connectance, num species)
- To characterize structure: cumulative degree distribution (fit to: powerlaw, exponencial, uniforme)
- Also look at: modularity (will return value of the intensity of the modularity of the total web), num of modules (max optimizer method idk)

Then, extinction analysis:
- KSI for bare-bones ranking
- extinction by KSI to validate
- top down / bottom up / wasp waist (use to confirm importance of species); works with trophic level of nodes
- When/if does the network collapse, and at what point / with which extinctions
    to measure the fragility of the network with respect to an ordered extinction
    secondary extinctions don't necessarily mean collapse, but they do mean a change in the network
    look at both
    
To evaluate importance of a node:
 - conjunction of KSI and identified module role
 - make this evaluation at each step of the extinction process to see how KSI changes for species, and what the value is at the point of extinction
 

"""


"""
Tomorrow's to-do:


"""


"""
Notes:

In static extinction, there may be redundant extinctions, i.e. extinguishing something that may have already been extinguished as a secondary effect
In dynamic extinction, there are no redundant extinctions. The new extrema are recalculated and popped, so if a species was secondarily extinguished in a prior iteration, it is gone.



"""