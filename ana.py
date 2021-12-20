import time
import heapq
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker,  draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import random

from astar import AstarSearch

class ANASearch(AstarSearch):

    def _put_new_nodes(self, current_node):
        self.visited_list[current_node.get_config()] = current_node.total_cost
        new_nodes = self._generate_new_nodes(current_node)
        for node in new_nodes:
            if node.total_cost+node.heuristic < self.G:
                config = node.get_config()
                if config in self.open_list:
                    if node.total_cost>=self.open_list[config][2].total_cost:
                        continue
                if config in self.visited_list:
                    if node.total_cost>=self.visited_list[config]:
                        continue
                e = (self.G-node.total_cost)/(node.heuristic+0.001)
                self.open_list[config] = (-e, random.random(), node)

    def _update_open_list(self):
        for config, (_,_,node) in self.open_list.copy().items():
            if node.total_cost+node.heuristic >= self.G:
                self.open_list.pop(node.get_config())
            else:
                e = (self.G-node.total_cost)/(node.heuristic+0.001)
                self.open_list[config] = (-e, random.random(), node)

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
        self.open_list = {}
        self.visited_list = {}
        self.start_node = self.create_node(start_config)
        self.goal_node = self.create_node(self.goal_config)
        self.history = []
        self.G = 10000000
        self.E = 10000000
        solution_found = False
        self._print('start point:', start_config)
        self._print('goal point:', self.goal_config)
        if self.collision_fn(list(self.goal_config)):
            self._print('=== Invalid goal!! ===')
            disconnect()
            return

        # statics
        start_time = time.time()
        debug_time = time.time()
        final_node = None
        final_cost = 0
        history = []
        path = []

        # main loop
        self._put_new_nodes(self.start_node)
        while len(self.open_list)>0:
            # improve solution
            while len(self.open_list)>0:
                open_list_minheap = [(v,k) for k,v in self.open_list.items()]
                heapq.heapify(open_list_minheap)
                current_step = heapq.heappop(open_list_minheap)
                current_priority = current_step[0][0]
                current_node = current_step[0][2]
                current_e = -1 * current_priority
                # self._print(current_node.get_config(), current_priority)
                self.open_list.pop(current_node.get_config())
                if current_e < self.E:
                    self.E = current_e
                if self._is_goal(current_node):
                    self.G = current_node.total_cost
                    solution_found = True
                    final_node = current_node
                    final_cost = final_node.total_cost
                    history.append((final_node, final_cost, time.time() - start_time))
                    color = [max(1-0.2*(len(history)-1), 0) for i in range(3)]
                    path = self._draw_path(final_node, color=color, z=0.1+len(history)*0.1)
                    break
                else:
                    self._put_new_nodes(current_node)
                if time.time()-start_time > self.timeout:
                    self.open_list = {}
                    break
            if not time.time()-start_time > self.timeout:
                self._print('[ANA* {}] Solution Cost={} time={:.4f}'.format(
                    self.n_connected,
                    final_cost,
                    time.time() - start_time
                ))
                self._update_open_list()


        # print statics
        if solution_found:
            # Execute planned path
            self._print('{} Solution Found!'.format(len(history)))
            # if use_gui:
            #     execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
        else:
            self._print('No Solution Found')
        if self.interact:
            wait_if_gui()
        disconnect()
        return history 


if __name__ == '__main__':
    ana = ANASearch(n_connected=4, grid_size=[0.1, 0.1, np.pi/2])
    ana.search(use_gui=True)
