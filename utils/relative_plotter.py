import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.append("../")

# Parse the coordinate data
coords_str = """442140.674 4491927.237 2324.603 % LCC1 ch1
442157.637 4491910.080 2321.258 % LCC1 ch2
442140.279 4491903.003 2319.584 % LCC1 ch3
441434.894 4491792.511 2267.894 % LCC2 ch1
441436.311 4491773.548 2264.880 % LCC2 ch2
441416.792 4491776.866 2263.025 % LCC2 ch3
440959.654 4491707.107 2237.975 % LCC3 ch1
440958.603 4491690.519 2232.528 % LCC3 ch2
440939.411 4491691.003 2233.343 % LCC3 ch3"""

# Extract the coordinates
coords = []
for line in coords_str.strip().split('\n'):
    parts = line.split('%')
    values = [float(val) for val in parts[0].strip().split()]
    label = parts[1].strip()
    coords.append({
        'x': values[0],
        'y': values[1],
        'z': values[2],
        'label': label
    })

# Filter out LCC1, keep only LCC2 and LCC3
lcc2_coords = [c for c in coords if c['label'].startswith('LCC2')]
lcc3_coords = [c for c in coords if c['label'].startswith('LCC3')]

# Calculate LCC2 center as origin
lcc2_center = {
    'x': sum(c['x'] for c in lcc2_coords) / len(lcc2_coords),
    'y': sum(c['y'] for c in lcc2_coords) / len(lcc2_coords),
    'z': sum(c['z'] for c in lcc2_coords) / len(lcc2_coords)
}

# Calculate relative coordinates
lcc2_relative = []
for c in lcc2_coords:
    lcc2_relative.append({
        'x': c['x'] - lcc2_center['x'],
        'y': c['y'] - lcc2_center['y'],
        'z': c['z'] - lcc2_center['z'],
        'label': c['label']
    })

lcc3_relative = []
for c in lcc3_coords:
    lcc3_relative.append({
        'x': c['x'] - lcc2_center['x'],
        'y': c['y'] - lcc2_center['y'],
        'z': c['z'] - lcc2_center['z'],
        'label': c['label']
    })

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# Plot LCC3 array
ax1.set_title('LCC3 array', fontsize=14, weight="bold")
ax1.set_xlabel('False easting (m)', fontsize=12, weight="bold")
ax1.set_ylabel('False northing (m)', fontsize=12, weight="bold")

# Extract x and y values for plotting
lcc3_x = [c['x'] for c in lcc3_relative]
lcc3_y = [c['y'] for c in lcc3_relative]

# Plot LCC3 points
for i, (x, y, c) in enumerate(zip(lcc3_x, lcc3_y, lcc3_relative)):
    ax1.scatter(x, y, color='red', marker='^', s=100)
    ax1.text(x + 1, y, str(i+1), fontsize=10)

# Add reference cross
ax1.plot(-480, -80, 'r+', markersize=5)
ax1.text(0, 1.1, 'B', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# Set limits for LCC3 plot
ax1.set_xlim(-500, -460)
ax1.set_ylim(-100, -60)
# ax1.set_xticks(fontsize=14)
# ax1.set_yticks(fontsize=14)
ax1.axis('equal')
ax1.grid(True)
ax1.add_patch(Rectangle((-500, -100), 40, 40, fill=False, edgecolor='black'))

# Plot LCC2 array
ax2.set_title('LCC2 array', fontsize=14, weight="bold")
ax2.set_xlabel('False easting (m)', fontsize=12, weight="bold")
ax2.set_ylabel('False northing (m)', fontsize=12, weight="bold")

# Extract x and y values for plotting
lcc2_x = [c['x'] for c in lcc2_relative]
lcc2_y = [c['y'] for c in lcc2_relative]

# Plot LCC2 points
for i, (x, y, c) in enumerate(zip(lcc2_x, lcc2_y, lcc2_relative)):
    ax2.scatter(x, y, color='red', marker='^', s=100)
    ax2.text(x + 1, y, str(i+1), fontsize=10)

# Add origin cross
ax2.plot(0, 0, 'r+', markersize=10)
ax2.text(0, 1.1, 'C', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# Set limits for LCC2 plot
ax2.set_xlim(-20, 20)
ax2.set_ylim(-20, 20)
# ax2.set_xticks(fontsize=14)
# ax2.set_yticks(fontsize=14)
ax2.axis('equal')
ax2.grid(True)
ax2.add_patch(Rectangle((-20, -20), 40, 40, fill=False, edgecolor='black'))



# plt.axis('equal')
plt.tight_layout()

plt.savefig(f"../plots/lcc2_lcc3_relative.png", dpi=300)
plt.show()