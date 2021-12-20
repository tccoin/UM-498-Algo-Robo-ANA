import numpy as np
import json
import copy

class PybulletPlayground:
    def __init__(self, template_path):
        with open(template_path) as template_json:
            self.template = json.load(template_json)
        self.table_template = [x for x in self.template['bodies'] if x['name']=='ikeatable1'][0]
        self.template['bodies'] = self.template['bodies'][:-1]

    def _euler_to_quaternion(self, yaw, pitch, roll):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return {"w":qw, "x":qx, "y":qy, "z":qz}
    
    def _to_pybullet_config(self, source, config):
        obj = copy.deepcopy(source)
        obj['point']['x'] = config[0]
        obj['point']['y'] = config[1]
        obj['quat'] = self._euler_to_quaternion(config[2],0,0)
        return obj

    def generate(self, output_path, floor_size, obstacle_config):
        playground = copy.deepcopy(self.template)
        # floor
        w,h = floor_size
        floor = playground['bodies'][0]
        floor['aabb']['extents'][0] = w/2+0.1
        floor['aabb']['extents'][1] = h/2+0.1
        floor['links'][0][0]['extents'][0] = w/2
        floor['links'][0][0]['extents'][1] = h/2

        floor['links'][0][1]['extents'][1] = h/2-0.2
        floor['links'][0][1]['point']['x'] = w/2-0.1
        floor['links'][0][2]['extents'][1] = h/2-0.2
        floor['links'][0][2]['point']['x'] = -1*(w/2-0.1)
        floor['links'][0][3]['extents'][0] = w/2
        floor['links'][0][3]['point']['y'] = h/2-0.1
        floor['links'][0][4]['extents'][0] = w/2
        floor['links'][0][4]['point']['y'] = -1*(h/2-0.1)
        # obstacle
        for i, config in enumerate(obstacle_config):
            table = self._to_pybullet_config(self.table_template, config)
            table['name'] = 'ikeatable{}'.format(i)
            playground['bodies'].append(table)
        with open(output_path, 'w') as playground_json:
            playground_json.write(json.dumps(playground))

if __name__ == '__main__':
    playground = PybulletPlayground('pr2playground_template.json')
    playground.generate(
        'pr2playground.json',
        floor_size=(8, 4),
        obstacle_config=[
            (-2, -1.2, 0),
            (0, 0.3, 0),
            (2, 1.2, 0)
        ]
    )
