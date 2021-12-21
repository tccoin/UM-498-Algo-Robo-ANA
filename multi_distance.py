from ana import ANASearch
from astar import AstarSearch
import numpy as np
from playground_generator import PybulletPlayground
import random
import multiprocessing as mp
from node import Node, Node2D, ManhattanDistanceNode, ManhattanDistanceNode2D, DiagonalDistanceNode, DiagonalDistanceNode2D

def search(task_index):
    # set searching arguments
    args = {
        'n_connected': 4 if task_index<6 else 8,
        'grid_size': [0.1, 0.1, np.pi/2],
        'start_config': (-x_max+0.5, 0, np.pi/2),
        'goal_config': (x_max-1, 0, -np.pi/2),
        'timeout': 1800,
        'camera_distance': 8,
        'angle_disabled': True,
        'verbose': False,
        'interact': False
    }
    node_class_list = [
        [Node, Node2D],
        [ManhattanDistanceNode, ManhattanDistanceNode2D],
        [DiagonalDistanceNode, DiagonalDistanceNode2D]
    ]
    node_class = node_class_list[(task_index%6)//2][args['angle_disabled']]
    args['create_node'] = node_class

    # conduct search
    ana = ANASearch(**args)
    astar = AstarSearch(**args)

    print('[{} {} {}] Start'.format(
        ['ANA*','A*'][task_index%2],
        args['n_connected'],
        node_class.__name__
    ))
    history = []
    if task_index%2==0:
        history = ana.search(use_gui=True, map=playground_filename)
    else:
        history = astar.search(use_gui=False, map=playground_filename)
    print('[{} {} {}]\ncost={}\ntime={}'.format(
        ['ANA*','A*'][task_index%2],
        args['n_connected'],
        node_class.__name__,
        [x[1] for x in history],
        [x[2] for x in history],
    ))

if __name__ == '__main__':
    # generate random scene
    random.seed(19774)
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
    obstacle_config = [c for c in obstacle_config if not (np.linalg.norm([c[0]+x_max,c[1]])<margin or np.linalg.norm([c[0]-x_max,c[1]])<margin)]
    playground = PybulletPlayground('pr2playground_template.json')
    playground_filename = 'pr2playground.json'
    playground.generate(playground_filename, floor_size, obstacle_config)

    # clac cost and time
    process_list = []
    # for i in range(12):
    for i in [6,8,10]:
        p = mp.Process(target=search, args=(i,))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()