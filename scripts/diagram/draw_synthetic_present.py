import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set font types for PDF/PS output
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Data: Speed-up ratios for different applications
applications = ['Multiphysics', 'N-Body', 'Scientific\nML', 'Data\nAssimilation']
speed_up_ratios = [0.704, 0.699, 0.35, 0.398] # With VLCs
baseline_ratios = [1.0, 1.0, 1.0, 1.0]     # Baseline (Without VLCs)

# Colors
colors_speedup = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
color_baseline = 'orange' # Grey color for baseline

# Set up bar positions
n_apps = len(applications)
x = np.arange(n_apps)  # the label locations
width = 0.35  # the width of the bars

# Create the figure and axes
plt.figure(figsize=(8, 4)) # Adjusted figure size for grouped bars

# Plot the bars
bars1 = plt.bar(x - width/2, baseline_ratios, width, label='Baseline (No VLCs)', color=color_baseline, edgecolor='black')
bars2 = plt.bar(x + width/2, speed_up_ratios, width, label='With VLCs', color=colors_speedup, edgecolor='black')

# Add data value labels to each bar
plt.bar_label(bars1, fmt='%.2f', padding=3, fontsize=12)
plt.bar_label(bars2, fmt='%.2f', padding=3, fontsize=12)

# Add titles and labels
plt.ylabel('Normalized Execution Time', fontsize=14)

# Set x-axis ticks and labels
plt.xticks(x, applications, rotation=0, fontsize=12)

# Add legend
plt.legend(loc="upper right") # Changed from "upper left" to "upper right"

# Remove the top and right borders (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True) # Ensure grid is behind bars

# Adjust layout and save the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('synthetic_present.pdf', format="pdf", bbox_inches='tight')  # Save the plot