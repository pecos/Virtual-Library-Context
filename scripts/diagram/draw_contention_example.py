import matplotlib.pyplot as plt
import numpy as np

# --- Data Estimated from the Original Image ---
# These values are approximations based on the provided bar chart.
val_openblas = 3287
val_openmp = 3439
val_default = 10450
# val_hand_tuned = 7000
val_vlc = 6433


# --- Create the Plot ---
plt.figure(figsize=(14, 6))

# Set the desired slim height for the bars
bar_height = 0.5 

# Plot the original bars (using the default blue color, 'C0')
bar_color = 'C0'

# --- Plot the New Stacked Bar ---
# This is the key part. We plot two bars at the same 'y' category.
# 1. Plot the 'OpenBLAS alone' component
plt.barh(
    'Sequential\n(OpenBLAS + OpenMP)', 
    val_openblas, 
    label='OpenBLAS alone (component)', 
    color='C1', # Use a different color (e.g., orange)
    height=bar_height # <-- Added height
)
# 2. Plot the 'OpenMP alone' component on top, using the 'left' parameter
plt.barh(
    'Sequential\n(OpenBLAS + OpenMP)', 
    val_openmp, 
    left=val_openblas, # This stacks it on top of the first component
    label='OpenMP alone (component)', 
    color='C2', # Use another different color (e.g., green)
    height=bar_height # <-- Added height
)

# Plot the other bars
plt.barh(
    'Parallel\n(Contention)', 
    val_default, 
    color=bar_color, 
    height=bar_height # <-- Added height
)
# plt.barh(
#     'Parallel (Hand-tuned)', 
#     val_hand_tuned, 
#     color=bar_color, 
#     height=bar_height # <-- Added height
# )
plt.barh(
    'VLCs', 
    val_vlc, 
    color=bar_color, 
    height=bar_height # <-- Added height
)

# --- Customize the Chart ---

# Invert the y-axis so 'OpenBLAS alone' is at the top, matching the original image
plt.gca().invert_yaxis()

# Add labels and a title with larger fonts
plt.xlabel('Execution Time (ms)', fontsize=22) # Increased from 12

# Increase tick font sizes
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Add a legend to explain the new stacked bar's colors
# plt.legend() # <-- Removed legend as requested

# Add a grid on the x-axis for easier reading
# plt.grid(axis='x', linestyle='--', alpha=0.7) # <-- Removed grid as requested

# Ensure all labels fit within the figure
plt.tight_layout()

# --- Save the Figure ---
plt.savefig('contention_example.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file