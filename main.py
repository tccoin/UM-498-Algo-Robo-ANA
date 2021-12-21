from ana import ANASearch
from astar import AstarSearch
import numpy as np
from playground_generator import PybulletPlayground
import random

seed = 19774
# seed = random.randint(1,1000)
random.seed(seed)
print(seed)

# generate random scene
# x_max, y_max, x_step, y_step, x_noise, y_noise = (10, 4, 2, 2, 1, 1)
x_max, y_max, x_step, y_step, x_noise, y_noise = (10, 10, 2, 2, 1, 1)
floor_size = (x_max*2, y_max*2)
obstacle_config = [
    (
        i+(random.random()-0.5)*2*x_noise,
        j+(random.random()-0.5)*2*y_noise,
        random.random()*np.pi
    )
    for i in np.arange(-x_max+x_step, x_max, x_step)
    for j in np.arange(-y_max+y_step, y_max, y_step)
]
margin = 3
# obstacle_config = [c for c in obstacle_config if not (c[0]<-x_max+margin and c[1]<-y_max+margin)]
# obstacle_config = [c for c in obstacle_config if not (c[0]>x_max-margin and c[1]>y_max-margin)]
obstacle_config = [c for c in obstacle_config if not (np.linalg.norm([c[0]+x_max,c[1]])<margin or np.linalg.norm([c[0]-x_max,c[1]])<margin)]
playground_filename = 'pr2playground.json'
playground = PybulletPlayground('pr2playground_template.json')
playground.generate(playground_filename, floor_size, obstacle_config)

# set searching arguments
args = {
    'n_connected': 8,
    'grid_size': [0.1, 0.1, np.pi/2],
    'start_config': (-x_max+0.5, 0, np.pi/2),
    'goal_config': (x_max-1, 0, -np.pi/2),
    'timeout': 1800,
    'camera_distance': 8,
    'angle_disabled': False,
    'verbose': True,
    'interact': True
}

# conduct search
ana = ANASearch(**args)
astar = AstarSearch(**args)

print('Searching {}...'.format(playground_filename))
ana.search(use_gui=True, map=playground_filename)
astar.search(use_gui=True, map=playground_filename)
