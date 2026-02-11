import matplotlib.pyplot as plt 
import numpy as np 
from GirdWorld import GridWorld


# use to visualize the grids with maatplotlib
for i in range(40,50):
    gw = GridWorld()
    gw.load(f'grids/grid_{i:02d}.pkl')

    blocked_pct = 100 * np.sum(gw.grid)/gw.size**2

    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(gw.grid, cmap='binary', interpolation='nearest')
    ax.set_title(f'Grid{i} ({blocked_pct:.1f} %blocked)', fontsize=16)
    ax.axis('off')

    plt.savefig(f'grid_images/grid_{i:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()


