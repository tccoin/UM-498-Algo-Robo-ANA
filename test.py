from ana import ANASearch
from astar import AstarSearch
import numpy as np
grid_size = [0.1, 0.1, np.pi/2]
goal_config = (2.6, 1.4, -np.pi/2)
astar = AstarSearch(n_connected=8, grid_size=grid_size,
                    goal_config=goal_config)
ana = ANASearch(n_connected=8, grid_size=grid_size,
                goal_config=goal_config, timeout=120)

# astar.search(use_gui=True)
# ana.search(use_gui=True)

ana.search(use_gui=False)
astar.search(use_gui=False)
