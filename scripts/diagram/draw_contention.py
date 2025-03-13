import matplotlib.pyplot as plt

# Data
configurations = ['With VLCs', '12 Threads\n Per Task', '24 Threads\n Per Task \n(Default)']
execution_times = [0.51, 0.66, 1.00]
color = '#1f77b4'  # Professional blue color

# Create the horizontal bar chart
plt.figure(figsize=(7, 4))
bars = plt.barh(configurations, execution_times, color=color, edgecolor='black', height=0.4)

# Add labels to bars
plt.bar_label(bars, fmt='%.2f', padding=3, fontsize=16)

# Add axis labels
plt.xlabel('Normalized Computation Times', fontsize=16)
# plt.ylabel('Configuration', fontsize=12)
plt.yticks(fontsize=16)

# Adjust x-axis limits for better spacing
plt.xlim(0, 1.1)

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# Add gridlines for clarity
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.savefig('contention.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file

plt.show()