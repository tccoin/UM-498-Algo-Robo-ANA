import numpy as np

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
        theta_diff = theta_diff if self.__class__.__name__.find('2D') == -1 else 0
        return np.sqrt(
            (self.x-other.x)**2
            + (self.y-other.y)**2
            + theta_diff**2
        )
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Node id {}: x={:.2f}, y={:.2f}, theta={:.2f}, parentid={}".format(
            self.id,
            self.x,
            self.y,
            self.theta,
            self.parent.id
        )

class Node2D(Node):
    def __init__(self, *args):
        super(Node2D, self).__init__(*args)

class ManhattanDistanceNode(Node):
    def __sub__(self, other):
        # cost for one step
        dx = abs(self.x-other.x)
        dy = abs(self.y-other.y)
        theta_diff = abs(self.theta - other.theta)
        theta_diff = min(np.pi*2-theta_diff, theta_diff)
        theta_diff = theta_diff if self.__class__.__name__.find('2D') == -1 else 0
        return dx+dy+theta_diff

class ManhattanDistanceNode2D(ManhattanDistanceNode):
    def __init__(self, *args):
        super(ManhattanDistanceNode2D, self).__init__(*args)

class DiagonalDistanceNode(Node):
    def __sub__(self, other):
        # cost for one step
        dx = abs(self.x-other.x)
        dy = abs(self.y-other.y)
        theta_diff = abs(self.theta - other.theta)
        theta_diff = min(np.pi*2-theta_diff, theta_diff)
        theta_diff = theta_diff if self.__class__.__name__.find('2D') == -1 else 0
        return dx+dy+(np.sqrt(2)-2)*min(dx,dy)+theta_diff


class DiagonalDistanceNode2D(DiagonalDistanceNode):
    def __init__(self, *args):
        super(DiagonalDistanceNode2D, self).__init__(*args)