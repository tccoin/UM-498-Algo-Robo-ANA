import time
import itertools
from queue import PriorityQueue
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker,  draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import pybullet as p

from node import Node, Node2D

class AstarSearch():
    def __init__(self, n_connected=4, grid_size=[0.1, 0.1, np.pi/2],start_config=(-9, -7, np.pi/2), goal_config=(9, 7, -np.pi/2), timeout=30, camera_distance=10, angle_disabled=False, create_node=None, verbose=True):
        if create_node is None:
            create_node = Node2D if angle_disabled else Node
        self.start_config = start_config
        self.goal_config = goal_config
        self.n_connected = n_connected
        self.grid_size = grid_size
        self.timeout = timeout
        self.camera_distance = camera_distance
        self.angle_disabled = angle_disabled
        self.create_node = create_node
        self.verbose = verbose

    def _angle_clip(self, angle):
        if angle >= np.pi:
            return angle - 2*np.pi
        elif angle <= -np.pi:
            return angle + 2*np.pi
        else:
            return angle

    def _generate_new_nodes(self, node):
        global collision_free_configs, colliding_configs
        new_configs = []

        config = list(node.get_config())
        if self.n_connected == 4:
            new_configs = [config.copy() for i in range(6)]
            new_configs[0][0] += self.grid_size[0]
            new_configs[1][0] -= self.grid_size[0]
            new_configs[2][1] += self.grid_size[1]
            new_configs[3][1] -= self.grid_size[1]
            new_configs[4][2] = self._angle_clip(config[2] + self.grid_size[2])
            new_configs[5][2] = self._angle_clip(config[2] - self.grid_size[2])
            if self.angle_disabled:
                new_configs = new_configs[:4]
        elif self.n_connected == 8:
            x, y, theta = config
            components = [
                (x, x+self.grid_size[0], x-self.grid_size[0]),
                (y, y+self.grid_size[1], y-self.grid_size[1]),
                (theta, theta+self.grid_size[2], theta-self.grid_size[2])
            ]
            if self.angle_disabled:
                components = components[:2]+[(0,)]
            product = itertools.product(*components)
            new_configs = [tuple(item) for item in product][1:]
        new_nodes = []
        for config in new_configs:
            if not self.collision_fn(config):
                new_nodes += [self.create_node(config, node, self.goal_node)]
        return new_nodes

    def _draw_path(self, last_node, color=None, z=0.2):
        path = [last_node.get_config()]
        while last_node.parent is not None:
            last_node = last_node.parent
            path += [last_node.get_config()]
        path.reverse()
        if color is None:
            color = (0, 0, 0)
        last_point = None
        for config in path:
            point = list(config)
            point[2] = z
            if last_point is not None:
                draw_line(last_point, point, 10, color)
            last_point = point
        return path

    def _put_new_nodes(self, current_node):
        new_nodes = self._generate_new_nodes(current_node)
        for node in new_nodes:
            if not node.get_config() in self.close_list:
                self.open_list.put(
                    (node.total_cost+node.heuristic, node.id, node))
                self.close_list[node.get_config()] = 1

    def _is_goal(self, node):
        return node.heuristic < self.grid_size[0]

    def _print(self, *args):
        if self.verbose:
            print(*args)

    def _set_camera(self):
        p.resetDebugVisualizerCamera(self.camera_distance, 0, -89.99, [0,0,0])

    def search(self, use_gui=True, map='pr2playground.json'):
        # init PyBullet
        connect(use_gui=use_gui)
        robots, obstacles = load_env(map)
        base_joints = [joint_from_name(robots['pr2'], name)
                       for name in PR2_GROUPS['base']]
        self.collision_fn = get_collision_fn_PR2(
            robots['pr2'], base_joints, list(obstacles.values()))
        self._set_camera()

        # init states
        start_config = tuple(self.start_config)
        goal_config = (2.6, -1.3, -np.pi/2)
        self.open_list = PriorityQueue()
        self.close_list = {}
        self.start_node = self.create_node(start_config)
        self.goal_node = self.create_node(self.goal_config)
        solution_found = False

        # statics
        start_time = time.time()
        final_node = None
        final_cost = 0
        path = []

        # main loop
        self._put_new_nodes(self.start_node)
        while not self.open_list.empty():
            current_step = self.open_list.get()
            current_node = current_step[2]
            if self._is_goal(current_node):
                solution_found = True
                final_node = current_node
                final_cost = final_node.total_cost
                break
            else:
                self._put_new_nodes(current_node)

        # print statics
        if solution_found:
            self._print('[A* {}]   Solution Cost={:.4f} time={:.4f}'.format(
                self.n_connected,
                final_cost,
                time.time() - start_time
            ))
            # draw path
            path = self._draw_path(final_node)
            # Execute planned path
            if use_gui:
                execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
        else:
            self._print('No Solution Found')
        disconnect()
        return 1 if solution_found else 0


if __name__ == '__main__':
    astar = AstarSearch(n_connected=8, grid_size=[0.6, 0.1, np.pi/2])
    astar.search(use_gui=False)
