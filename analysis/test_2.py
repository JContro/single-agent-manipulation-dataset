import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
group_labels = ['Group 1', 'Group 2', 'Group 3']

# Data for each group (means)
group1_means = [10, 12, 8, 11]
group2_means = [12, 15, 10, 13]
group3_means = [8, 10, 6, 9]

# Asymmetric errors for each group
# Format: [lower_errors, upper_errors]
group1_errors = [[5, 1.8, 1.2, 1.4],   # lower bounds
                 [0.8, 1.2, 0.9, 1.1]]    # upper bounds
group2_errors = [[2.0, 2.2, 1.5, 1.8],
                 [1.2, 1.5, 1.0, 1.3]]
group3_errors = [[1.2, 1.5, 1.0, 1.1],
                 [0.9, 1.1, 0.7, 0.8]]

# Set width of bars and positions of the bars
bar_width = 0.25
x = np.arange(len(categories))

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for each group with asymmetric error bars
bars1 = ax.bar(x - bar_width, group1_means, bar_width, 
               yerr=group1_errors,  # This takes [lower, upper] arrays
               label=group_labels[0],
               capsize=5,
               color='skyblue',
               ecolor='black')

bars2 = ax.bar(x, group2_means, bar_width,
               yerr=group2_errors,
               label=group_labels[1],
               capsize=5,
               color='lightgreen',
               ecolor='black')

bars3 = ax.bar(x + bar_width, group3_means, bar_width,
               yerr=group3_errors,
               label=group_labels[2],
               capsize=5,
               color='salmon',
               ecolor='black')

# Customize the plot
ax.set_ylabel('Values')
ax.set_xlabel('Categories')
ax.set_title('Grouped Bar Plot with Asymmetric Error Bars')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('grouped_bar_plot_asymmetric_errors.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
