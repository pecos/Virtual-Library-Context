import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import textwrap

# Data: Speed-up ratios for different applications
applications = ['Kmeans & Hotspot3D', 'Kmeans & CFD', 'Hotspot3D & CFD', 'Kmeans & Hotspot3D (ARM)', 'Kmeans & CFD (ARM)', 'Hotspot3D & CFD (ARM)', 'Cholesky & GEMM', 'Cholesky & GESV', 'GEMM & GESV', 'GPTlite & Deep DNN', 'GPTlite & Wide DNN', 'Deep DNN & Wide DNN', 'Hotspot3D & Cholesky', 'Cholesky & Deep DNN', 'Hotspot3D & Deep DNN']
speed_up_ratios = [1.57,1.72,1.46,1.28,1.32,1.25,1.35,1.41,1.17,1.48,1.62,1.94,1.38,2.05,3.37]

wrapped_labels = ['\n'.join(textwrap.wrap(label, width=11)) for label in applications]

colors = ['#1f77b4'] * 3 + ['#d62728'] * 3 + ['#ff7f0e'] * 3 + ['#2ca02c'] * 3 + ['grey'] * 3

# Create the histogram
plt.figure(figsize=(15, 6))
bars = plt.bar(wrapped_labels, speed_up_ratios, color=colors, edgecolor='black', width=0.4)

# Create legend using proxy artists
legend_handles = [
    Patch(color='#1f77b4', label='OpenMP'),
    Patch(color='#d62728', label='OpenMP (ARM)'),
    Patch(color='#ff7f0e', label='OpenBLAS'),
    Patch(color='#2ca02c', label='LibTorch'),
    Patch(color='grey', label='Mixed')
]
plt.legend(handles=legend_handles, title="Groups", loc="upper left", fontsize=14)

# Add data value labels to each bar
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=14)  # Format values to one decimal place

# Add titles and labels
# plt.title('Histogram of Speed Up Ratios Across Benchmarks Each Composes 2 Tasks', fontsize=16)
# plt.xlabel('Benchmarks', fontsize=14)
plt.ylabel('Speed Up Ratio', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=35, fontsize=14)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('speedup_2tasks.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap