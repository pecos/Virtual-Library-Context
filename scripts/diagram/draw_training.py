import matplotlib.pyplot as plt

instances = ["1", "2", "4"]
vlcs_speedup = [1.00, 3.22, 6.43]
base_speedup = [1.00, 2.05, 2.46]
best_speedup = [1.00, 2.76, 4.76]

plt.figure(figsize=(5, 4))
plt.plot(instances, vlcs_speedup, marker='o', label='With VLCs', color='blue')
plt.plot(instances, base_speedup, marker='o', label='Without VLCs (Default)', color='red')
plt.plot(instances, best_speedup, marker='o', label='Without VLCs (Best Config)', color='green')

x_labels = ['1', '2', '4']
plt.xticks(ticks=instances, labels=x_labels)

for i in range(len(instances)):
    if i == 1:
        plt.text(instances[i], vlcs_speedup[i] + 0.2, f'{vlcs_speedup[i]:.2f}', color='blue', fontsize=8, ha='center', va='bottom')
    else:
        plt.text(instances[i], vlcs_speedup[i] + 0.1, f'{vlcs_speedup[i]:.2f}', color='blue', fontsize=8, ha='center', va='bottom')
    plt.text(instances[i], base_speedup[i] + 0.1, f'{base_speedup[i]:.2f}', color='red', fontsize=8, ha='center', va='bottom')
    plt.text(instances[i], best_speedup[i] + 0.1, f'{best_speedup[i]:.2f}', color='green', fontsize=8, ha='center', va='bottom')


plt.xlabel('Number of Tasks Run Simultaneously')
plt.ylabel('Speedup Ratio')
plt.legend()

# Remove the border (spines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()  # Adjusts layout to prevent label overlap
plt.savefig('training.pdf', format="pdf", bbox_inches='tight')  # To save the heatmap as an image file
plt.show()  # To display the heatmap