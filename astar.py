import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from utils import draw_sphere_marker, draw_line
from queue import PriorityQueue
import itertools
#########################


def main(screenshot=False, n_connected_neighbors=4, grid_size=[0.1, 0.1, np.pi/2]):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name)
                   for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(
        robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###

    def angle_clip(angle):
        if angle >= np.pi:
            return angle - 2*np.pi
        elif angle <= -np.pi:
            return angle + 2*np.pi
        else:
            return angle

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
            theta_diff = abs(self.theta - other.theta)
            return np.sqrt(
                (self.x-other.x)**2
                + (self.y-other.y)**2
                + (min(np.pi*2-theta_diff, theta_diff))**2
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

    def generate_new_nodes(node):
        global collision_free_configs, colliding_configs
        new_configs = []

        config = list(node.get_config())
        if n_connected_neighbors == 4:
            new_configs = [config.copy() for i in range(6)]
            new_configs[0][0] += grid_size[0]
            new_configs[1][0] -= grid_size[0]
            new_configs[2][1] += grid_size[1]
            new_configs[3][1] -= grid_size[1]
            new_configs[4][2] = angle_clip(new_configs[4][2] + grid_size[2])
            new_configs[5][2] = angle_clip(new_configs[5][2] - grid_size[2])
        elif n_connected_neighbors == 8:
            x, y, theta = config
            product = itertools.product(
                (x, x+grid_size[0], x-grid_size[0]),
                (y, y+grid_size[1], y-grid_size[1]),
                (theta, theta+grid_size[2], theta-grid_size[2]),
            )
            new_configs = [tuple(item) for item in product][1:]
        new_nodes = []
        collision_free_configs = []
        colliding_configs = []
        for config in new_configs:
            if not collision_fn(config):
                new_nodes += [Node(config, node, goal_node)]
                collision_free_configs += [config]
            else:
                colliding_configs += [config]
        return new_nodes, collision_free_configs, colliding_configs

    def put_new_nodes(current_node):
        new_nodes, collision_free_configs, colliding_configs = generate_new_nodes(
            current_node)
        for config in collision_free_configs:
            collision_free_positions[tuple(config[:2])] = 1
        for config in colliding_configs:
            colliding_positions[tuple(config[:2])] = 1
        for node in new_nodes:
            if not node.get_config() in close_list:
                open_list.put((node.total_cost+node.heuristic, node.id, node))
                close_list[node.get_config()] = 1

    open_list = PriorityQueue()
    close_list = {}
    collision_free_positions = {}
    colliding_positions = {}
    start_node = Node(start_config)
    goal_node = Node(goal_config)

    final_node = None
    final_cost = 0
    solution_found = False

    put_new_nodes(start_node)
    while not open_list.empty():
        current_step = open_list.get()
        current_node = current_step[2]
        # print(current_node)
        if current_node.heuristic < 0.01:
            solution_found = True
            final_node = current_node
            final_cost = final_node.total_cost
            break
        else:
            put_new_nodes(current_node)

    path = [final_node.get_config()]
    while final_node.parent is not None:
        final_node = final_node.parent
        path += [final_node.get_config()]
    path.reverse()

    if solution_found:
        print('Solution Found!!!')
        print('\tUsing {}-connected nerighbors'.format(n_connected_neighbors))
        print('\tPath cost: {:.4f}'.format(final_cost))
        print("\tPlanner run time: ", time.time() - start_time)

        # draw path
        last_point = None
        for config in path:
            point = list(config)
            point[2] = 0.2
            if last_point is not None:
                draw_line(last_point, point, 10, (0, 0, 0))
            last_point = point

        # draw configs
        print(len(collision_free_positions), len(colliding_positions))
        for config in list(collision_free_positions.keys()):
            position = list(config)
            position += [0.1]
            draw_sphere_marker(position, 0.03, (0, 0, 1, 1))
        for config in list(colliding_positions.keys()):
            position = list(config)
            position += [0.1]
            draw_sphere_marker(position, 0.03, (1, 0, 0, 1))
    else:
        print('No Solution Found')

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    # main(n_connected_neighbors=4)
    main(n_connected_neighbors=8)
