from ana import ANASearch
from astar import AstarSearch
import numpy as np
from playground_generator import PybulletPlayground
import random
import multiprocessing as mp

def search(i):
    path_len = 0

    while True:
        seed = random.randint(1,100000)
        random.seed(seed)
        print('[Start: Seed={}]'.format(seed))

        # generate random scene
        # x_max, y_max, x_step, y_step, x_noise, y_noise = (8, 4, 1.5, 1.5, 1, 1)
        x_max, y_max, x_step, y_step, x_noise, y_noise = (5, 3, 1.5, 1.5, 1, 1)
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
            'timeout': 600,
            'camera_distance': 10,
            'angle_disabled': False,
            'verbose': False
        }

        # conduct search
        ana = ANASearch(**args)
        astar = AstarSearch(**args)

        history_ana = ana.search(use_gui=False, map=playground_filename)
        path_len = len(history_ana)
        if path_len>2:
            print('[Found: Seed={}, path={}]'.format(seed, path_len))
            return
        elif path_len>0:
            print('[Stop:  Seed={}, path={}]'.format(seed, path_len))

if __name__ == '__main__':
    process_list = []
    for i in range(mp.cpu_count()):
        p = mp.Process(target=search, args=(i,))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()