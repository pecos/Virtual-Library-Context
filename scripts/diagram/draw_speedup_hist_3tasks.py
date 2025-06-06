import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import textwrap

# Data: Speed-up ratios for different applications
applications = ['Kmeans & Hotspot3D & CFD', 'Kmeans & Hotspot3D & CFD (ARM)', 'Cholesky & GEMM &  GESV', 'GPTlite & Deep DNN & Wide DNN', 'Hotspot3D & Cholesky & Deep DNN']
speed_up_ratios = [1.56, 1.36, 1.43, 1.93,2.24]

wrapped_labels = ['\n'.join(textwrap.wrap(label, width=11)) for label in applications]

colors = ['#1f77b4'] * 1 + ['#d62728'] * 1 + ['#ff7f0e'] * 1 + ['#2ca02c'] * 1 + ['grey'] * 1

# Create the histogram
plt.figure(figsize=(6, 6.5))
bars = plt.bar(wrapped_labels, speed_up_ratios, color=colors, edgecolor='black', width=0.4)

# Create legend using proxy artists
legend_handles = [
    Patch(color='#1f77b4', label='OpenMP'),
    Patch(color='#d62728', label='OpenMP (ARM)'),
    
    Patch(color='#ff7f0e', label='OpenBLAS'),
    Patch(color='#2ca02c', label='LibTorch'),
    Patch(color='grey', label='Mixed')
]
plt.legend(handles=legend_handles, title="Groups", loc="upper left")

# Add data value labels to each bar
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=14)  # Format values to one decimal place

# Add titles and labels
# plt.title('Histogram of Speed Up Ratios Across Benchmarks\nEach Composes 3 Tasks', fontsize=16)
# plt.xlabel('Benchmarks', fontsize=14)
plt.ylabel('Speed Up Ratio', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=35, fontsize=12)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('speedup_3tasks.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap