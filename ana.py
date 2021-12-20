import time
import itertools
from queue import PriorityQueue
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker,  draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import pybullet as p


class Node:
    global_id = 0

    def __init__(self, config_in, parent_in=None, goal_in=None):
        self.x = config_in[0]
        self.y = config_in[1]
        self.theta = config_in[2]
        self.id = self.global_id + 1
        Node.global_id += 1
        self.parent = parent_in
        if parent_in is not None:
            self.total_cost = parent_in.total_cost + (self - parent_in)
        else:
            self.total_cost = 0
        if goal_in is not None:
            self.heuristic = self - goal_in

    def get_config(self):
        return (self.x, self.y, self.theta)

    def __sub__(self, other):
        # cost for one step
        theta_diff = abs(self.theta - other.theta)
        theta_diff = min(np.pi*2-theta_diff, theta_diff)
        return np.sqrt(
            (self.x-other.x)**2
            + (self.y-other.y)**2
            + theta_diff**2
        )

    def __str__(self):
        return "\tNode id {}: x={:.2f}, y={:.2f}, theta={:.2f}, parentid={} g={:.2f} h={:.2f} f={:.2f}.".format(
            self.id,
            self.x,
            self.y,
            self.theta,
            self.parent.id,
            self.total_cost,
            self.heuristic,
            self.total_cost+self.heuristic
        )


class ANASearch():
    def __init__(self, n_connected=4, grid_size=[0.1, 0.1, np.pi/2],start_config=(-3.4, -1.4, np.pi/2), goal_config=(3.4, 1.4, -np.pi/2), timeout=30, camera_distance=3):
        self.start_config = start_config
        self.goal_config = goal_config
        self.n_connected = n_connected
        self.grid_size = grid_size
        self.timeout = timeout
        self.camera_distance = camera_distance

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
        elif self.n_connected == 8:
            x, y, theta = config
            product = itertools.product(
                (x, x+self.grid_size[0], x-self.grid_size[0]),
                (y, y+self.grid_size[1], y-self.grid_size[1]),
                (theta, theta+self.grid_size[2], theta-self.grid_size[2]),
            )
            new_configs = [tuple(item) for item in product][1:]
        new_nodes = []
        for config in new_configs:
            if not self.collision_fn(config):
                new_nodes += [Node(config, node, self.goal_node)]
        return new_nodes

    def _put_new_nodes(self, current_node):
        # self.close_list[current_node.get_config()] = 1
        new_nodes = self._generate_new_nodes(current_node)
        for node in new_nodes:
            if (node.total_cost+node.heuristic < self.G):
            # if (node.total_cost+node.heuristic < self.G) and (not node.get_config() in self.close_list):
                e = (self.G-node.total_cost)/(node.heuristic+0.001)
                print(node.get_config(),e,node.total_cost)
                self.open_list.put((-1*e, node.id, node))

    def _draw_path(self, last_node, color=None):
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
            point[2] = 0.2
            if last_point is not None:
                draw_line(last_point, point, 10, color)
            last_point = point

    def _update_open_list(self):
        node_map = {}
        while not self.open_list.empty():
            e, _, node = self.open_list.get()
            config = node.get_config()
            if node.total_cost+node.heuristic >= self.G:
                continue
            if (not config in node_map) or (node.total_cost < node_map[config].total_cost):
                node_map[config] = node
        self.open_list = PriorityQueue()
        for _, node in node_map.items():
            e = (self.G-node.total_cost)/node.heuristic
            self.open_list.put((-1*e, node.id, node))

    def _is_goal(self, node):
        return node.heuristic < self.grid_size[0]

    def _set_camera(self):
        p.resetDebugVisualizerCamera(self.camera_distance, 0, -89, [0,0,0])

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
        self.open_list = PriorityQueue()
        self.start_node = Node(start_config)
        self.goal_node = Node(self.goal_config)
        self.close_list = {}
        self.history = []
        self.G = 1000000
        self.E = 1000000
        solution_found = False
        print('start point:', start_config)
        print('goal point:', self.goal_config)
        if self.collision_fn(list(self.goal_config)):
            print('=== Invalid goal!! ===')
            disconnect()
            return

        # statics
        start_time = time.time()
        debug_time = time.time()
        final_node = None
        final_cost = 0
        history = []

        # main loop
        self._put_new_nodes(self.start_node)
        while not self.open_list.empty():
            # improve solution
            while not self.open_list.empty():
                current_step = self.open_list.get()
                current_priority, _, current_node = current_step
                current_e = -1 * current_priority
                # if(time.time()-debug_time > 1):
                #     debug_time = time.time()
                #     print(current_e)
                if current_e < self.E:
                    self.E = current_e
                if self._is_goal(current_node):
                    self.G = current_node.total_cost
                    solution_found = True
                    final_node = current_node
                    final_cost = final_node.total_cost
                    history.append((final_node, final_cost))
                    color = [max(0, 285-100*len(history)) for i in range(3)]
                    self._draw_path(final_node, color=())
                    break
                else:
                    self._put_new_nodes(current_node)
                if time.time()-start_time > self.timeout:
                    self.open_list = PriorityQueue()
                    break
            if not time.time()-start_time > self.timeout:
                print('[ANA* {}] Solution Cost={} time={:.4f}'.format(
                    self.n_connected,
                    final_cost,
                    time.time() - start_time
                ))
                self.close_list = {}
                self._update_open_list()


        # print statics
        if solution_found:
            print('Solution Found!')
            # draw path
            # self._draw_path(final_node)
            # Execute planned path
            if use_gui:
                execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
        else:
            print('No Solution Found')

        wait_if_gui()
        disconnect()


if __name__ == '__main__':
    ana = ANASearch(n_connected=4, grid_size=[0.1, 0.1, np.pi/2])
    ana.search(use_gui=True)
