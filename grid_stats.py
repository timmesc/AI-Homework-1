import numpy as np
from GirdWorld import GridWorld

percentages = []
for i in range(50):
    gw = GridWorld()
    gw.load(f'grids/grid_{i:02d}.pkl')
    pct = 100 * np.sum(gw.grid)/(gw.size **2)
    percentages.append(pct)

print(f'Mean: {np.mean(percentages):.2f}%')
print(f'Std Dev: {np.std(percentages):.2f}%')
print(f'Min: {np.min(percentages):.2f}%') 
print(f'Max: {np.max(percentages):.2f}%') 
