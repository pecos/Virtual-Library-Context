import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import textwrap

# Data: Speed-up ratios for different applications
applications = ['Kmeans & Hotspot3D', 'Kmeans & CFD', 'Hotspot3D & CFD', 'Cholesky & GEMM', 'Cholesky & Linear Equation Solver', 'GEMM & Linear Equation Solver', 'GPTlite & Deep DNN', 'GPTlite & Wide DNN', 'Deep DNN & Wide DNN', 'Hotspot3D & Cholesky', 'Cholesky & Deep DNN', 'Hotspot3D & Deep DNN']
speed_up_ratios = [1.74,1.92,3.31,1.57,1.58,1.44,3.78,4.33,1.98,1.43,4.88,4.26]

wrapped_labels = ['\n'.join(textwrap.wrap(label, width=10)) for label in applications]

colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 3 + ['#2ca02c'] * 3 + ['grey'] * 3

# Create the histogram
plt.figure(figsize=(12, 6))
bars = plt.bar(wrapped_labels, speed_up_ratios, color=colors, edgecolor='black', width=0.4)

# Create legend using proxy artists
legend_handles = [
    Patch(color='#1f77b4', label='OpenMP'),
    Patch(color='#ff7f0e', label='OpenBLAS'),
    Patch(color='#2ca02c', label='LibTorch'),
    Patch(color='grey', label='Mixed')
]
plt.legend(handles=legend_handles, title="Groups", loc="upper left")

# Add data value labels to each bar
plt.bar_label(bars, fmt='%.2f', padding=3)  # Format values to one decimal place

# Add titles and labels
plt.title('Histogram of Speed Up Ratios Across Benchmarks Each Composes 2 Tasks', fontsize=16)
plt.xlabel('Benchmarks', fontsize=14)
plt.ylabel('Speed Up Ratio', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=0,fontsize=10)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('speedup_2tasks.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap