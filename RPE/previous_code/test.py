import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Define the custom color map
colors = ["green", "lime", "white", "pink", "deeppink"]  # Corrected color name
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Create a figure and a axes
fig, ax = plt.subplots(figsize=(1.5, 4))

# Create a colorbar with the custom colormap
norm = plt.Normalize(vmin=-15, vmax=5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You have to set the array for the ScalarMappable
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', ticks=[-15, -10, -5, 0, 5])
cbar.ax.set_yticklabels(['-15', '-10', '-5', '0', '5'])

# Remove the axis
ax.remove()

plt.savefig('test')
