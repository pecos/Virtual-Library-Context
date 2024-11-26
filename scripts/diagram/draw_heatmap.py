import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "heatmap.csv"

def count(x):
    x_value = x.split('-')
    return abs(int(x_value[0]) - int(x_value[1])) + 1


# Load CSV data into a pandas DataFrame
df = pd.read_csv(CSV_FILE)

df['max_value'] = df[['value1', 'value2']].max(axis=1)
df['x_count'] = df['x'].apply(count)
df['y_count'] = df['y'].apply(count)

# Pivot the DataFrame to create a matrix suitable for heatmap plotting
heatmap_data = df.pivot(index='y_count', columns='x_count', values='max_value')

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='.2f')

# Highlight the diagonal elements with a rectangular frame
for i in range(len(heatmap_data)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

# Find the position of the minimum value in the DataFrame
min_val = heatmap_data.min().min()  # Find the minimum value
min_pos = np.where(heatmap_data == min_val)  # Get its position (row, col)
min_row, min_col = min_pos[0][0], min_pos[1][0]  # Extract row and column indices
ax.add_patch(plt.Rectangle((min_col, min_row), 1, 1, fill=False, edgecolor='green', lw=3))

# # Mark the default case
# ax.add_patch(plt.Rectangle((len(heatmap_data) - 1, len(heatmap_data) - 1), 1, 1, fill=False, edgecolor='orange', lw=4))

# Add titles and labels
plt.title('Heatmap of Resouce Parition between two DNN tasks', fontsize=16)
plt.xlabel('Number of CPU cores assigned to Deep DNN task', fontsize=14)
plt.ylabel('Number of CPU cores assigned to Wide DNN task', fontsize=14)

# Save the heatmap as an image file or display it
plt.savefig('heatmap.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap