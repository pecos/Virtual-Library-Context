import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import textwrap

# Data: Speed-up ratios for different applications
applications = ['Multiphysics', 'N-Body', 'Scientific\nML', 'Data\nAssimilation']
speed_up_ratios = [1.42, 1.43, 2.85, 2.51]

wrapped_labels = [label for label in applications]

colors = ['#1f77b4'] * 1 + ['#d62728'] * 1 + ['#ff7f0e'] * 1 + ['#2ca02c'] * 1

# Create the histogram
plt.figure(figsize=(6, 3))
bars = plt.bar(wrapped_labels, speed_up_ratios, color=colors, edgecolor='black', width=0.4)

# Create legend using proxy artists
# legend_handles = [
#     Patch(color='#1f77b4', label='OpenMP'),
#     Patch(color='#d62728', label='OpenMP (ARM)'),
    
#     Patch(color='#ff7f0e', label='OpenBLAS'),
#     Patch(color='#2ca02c', label='LibTorch'),
# ]
# plt.legend(handles=legend_handles, title="Groups", loc="upper left")

# Add data value labels to each bar
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=14)  # Format values to one decimal place

# Add titles and labels
# plt.title('Histogram of Speed Up Ratios Across Benchmarks\nEach Composes 3 Tasks', fontsize=16)
# plt.xlabel('Benchmarks', fontsize=14)
plt.ylabel('Speedup Ratio', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=0, fontsize=12)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(True)

# Display the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('synthetic.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap