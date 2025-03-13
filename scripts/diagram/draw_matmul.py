import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Data
categories = ['Sequential', 'Multithread', 'VLCs',
              'Sequential ', 'Multithread ', 'VLCs ']
execution_times = [1.00, 0.98, 0.83, 1.00, 0.74, 0.63]
colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 3

# Create the horizontal bar chart
plt.figure(figsize=(9, 4))
bars = plt.bar(categories, execution_times, color=colors, edgecolor='black', width=0.4)


# Create legend using proxy artists
legend_handles = [
    Patch(color='#1f77b4', label='Python'),
    Patch(color='#ff7f0e', label='C++')
]
plt.legend(handles=legend_handles, title="Groups", loc="upper right", fontsize=14)

# Add labels to bars
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=14)

# Add axis labels
plt.ylabel('Normalized Computation Times', fontsize=14)
# plt.ylabel('Configuration', fontsize=12)
plt.xticks(fontsize=14)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.savefig('matmul.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file

plt.show()