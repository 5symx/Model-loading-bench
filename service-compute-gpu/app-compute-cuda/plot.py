import matplotlib.pyplot as plt
import numpy as np

# Constants for the size of data transferred
size_of_int = 4  # size of int in bytes
MB = 1024 * 1024  # 1 MB in bytes

# Data for the plot
categories = ['memcpyD2H', 'memcpyH2D']
without_initialize_without_cuhost = [1.804533, 0.782502]
with_initialize_without_cuhost = [0.490478, 0.785875]
with_initialize_with_cuhost = [0.290768, 0.607828]
without_initialize_with_cuhost = [0.277467, 0.578984]

# Calculate the size of data transferred in MB
size_memcpyD2H_MB = (1024 * 1024 * size_of_int) / MB
size_memcpyH2D_MB = (1024 * 1024 * size_of_int * 2) / MB

# Normalize the durations per MB
def normalize_duration(durations, size_MB):
    return [duration / size_MB for duration in durations]

normalized_without_initialize_without_cuhost = [
    without_initialize_without_cuhost[0] / size_memcpyD2H_MB,
    without_initialize_without_cuhost[1] / size_memcpyH2D_MB
]
normalized_with_initialize_without_cuhost = [
    with_initialize_without_cuhost[0] / size_memcpyD2H_MB,
    with_initialize_without_cuhost[1] / size_memcpyH2D_MB
]
normalized_with_initialize_with_cuhost = [
    with_initialize_with_cuhost[0] / size_memcpyD2H_MB,
    with_initialize_with_cuhost[1] / size_memcpyH2D_MB
]
normalized_without_initialize_with_cuhost = [
    without_initialize_with_cuhost[0] / size_memcpyD2H_MB,
    without_initialize_with_cuhost[1] / size_memcpyH2D_MB
]

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*1.5, normalized_without_initialize_without_cuhost, width, label='Without Initialize Without CuHost')
rects2 = ax.bar(x - width/2, normalized_with_initialize_without_cuhost, width, label='With Initialize Without CuHost')
rects3 = ax.bar(x + width/2, normalized_with_initialize_with_cuhost, width, label='With Initialize With CuHost')
rects4 = ax.bar(x + width*1.5, normalized_without_initialize_with_cuhost, width, label='Without Initialize With CuHost')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Operation Name')
ax.set_ylabel('Normalized Duration (ms/MB)')
ax.set_title('Normalized Duration by Operation and Initialization/CuHost Status')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()

# Save the plot as a PNG file
plt.savefig('normalized_duration_plot.png')

print("The plot has been saved as 'normalized_duration_plot.png'.")