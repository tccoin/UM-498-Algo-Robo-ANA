from ana import ANASearch
from astar import AstarSearch
import numpy as np
from playground_generator import PybulletPlayground
import random

# init scene
x_max, y_max, x_step, y_step, x_noise, y_noise = (10, 8, 2, 2, 1, 1)
floor_size = (x_max*2, y_max*2)
obstacle_config = [
    (
        i+(random.random()-0.5)*2*x_noise,
        j+(random.random()-0.5)*2*y_noise,
        # (random.random()-0.5)*(y_max),
        random.random()*np.pi
    )
    for i in range(-x_max+x_step, x_max, x_step)
    for j in range(-y_max+y_step, y_max, y_step)
]
margin = 3
obstacle_config = [x for x in obstacle_config if not (x[0]<-x_max+margin and x[1]<-y_max+margin)]
obstacle_config = [x for x in obstacle_config if not (x[0]>x_max-margin and x[1]>y_max-margin)]
playground = PybulletPlayground('pr2playground_template.json')
playground.generate('pr2playground.json', floor_size, obstacle_config)
# playground.generate([
#         ,
#         (0, random.random()*2.4-1.2, random.random()*np.pi),
#         (2, random.random()*2.4-1.2, random.random()*np.pi),
#     ], 'pr2playground.json'
# )

# set searching config
args = {
    'n_connected':4,
    'grid_size': [0.3, 0.3, np.pi/2],
    'start_config': (-x_max+1, -y_max+1, np.pi/2),
    'goal_config': (x_max-1, y_max-1, -np.pi/2),
    'timeout':240,
    'camera_distance': 10
}

ana = ANASearch(**args)
astar = AstarSearch(**args)

ana.search(use_gui=True, map='pr2playground.json')
# astar.search(use_gui=True, map='pr2playground.json')

# ana.search(use_gui=False)
# astar.search(use_gui=False)
