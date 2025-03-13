import matplotlib.pyplot as plt
import numpy as np

# Data
# Data for the first graph
categories = ['C++', 'Python']
subcategories = ['Sequential', 'VLCs +\nDefault Partition', 'VLCs +\nEven Partition']
data = np.array([
    [1.00, 0.90, 0.51],  # C++
    [1.00, 1.03, 0.66]   # Python
])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create the bar chart
x = np.arange(len(categories))  # X positions for categories
width = 0.15  # Width of each bar
plt.figure(figsize=(6, 4))

# Create the horizontal bar chart
for i in range(len(subcategories)):
    bars = plt.bar(x + i * width, data[:, i], width, label=subcategories[i], color=colors[i], edgecolor='black')
    plt.bar_label(bars, fmt='%.2f', fontsize=10)

plt.legend(subcategories, fontsize=10, loc='upper center')

# Add labels to bars
# plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=14)

# Add axis labels
plt.ylabel('Normalized Computation Times', fontsize=14)
# plt.ylabel('Configuration', fontsize=12)
plt.xticks(x + width, categories, fontsize=12)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.savefig('arpack.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file

plt.show()