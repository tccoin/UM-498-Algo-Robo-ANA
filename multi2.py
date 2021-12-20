from ana import ANASearch
from astar import AstarSearch
import numpy as np
from playground_generator import PybulletPlayground
import random
import multiprocessing as mp

def search(seed):
    random.seed(seed)
    print('[Start: Seed={}]'.format(seed))

    # generate random scene
    # x_max, y_max, x_step, y_step, x_noise, y_noise = (8, 4, 1.5, 1.5, 1, 1)
    x_max, y_max, x_step, y_step, x_noise, y_noise = (10, 4, 2, 2, 1, 1)
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
    playground = PybulletPlayground('pr2playground_template.json')
    playground_filename = 'dist/pr2playground {}.json'.format(seed)
    playground.generate(playground_filename, floor_size, obstacle_config)

    # set searching arguments
    args = {
        'n_connected': 8,
        'grid_size': [0.1, 0.1, np.pi/2],
        'start_config': (-x_max+0.5, 0, np.pi/2),
        'goal_config': (x_max-1, 0, -np.pi/2),
        'timeout': 3600,
        'camera_distance': 8,
        'angle_disabled': True,
        'verbose': False,
        'interact': False
    }

    # conduct search
    ana = ANASearch(**args)
    astar = AstarSearch(**args)

    history_ana = ana.search(use_gui=False, map=playground_filename)
    history_astar = astar.search(use_gui=False, map=playground_filename)
    print('[Seed={}]\ncost_ana={}\ntime_ana={}\ncost_astar={}\n'.format(
        seed,
        [x[1] for x in history_ana],
        [x[2] for x in history_ana],
        [x[1] for x in history_astar],
        [x[2] for x in history_astar]
    ))

if __name__ == '__main__':
    process_list = []
    for seed in [54010,80105,51981,6045,19774,35556,49047]:
        p = mp.Process(target=search, args=(seed,))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()