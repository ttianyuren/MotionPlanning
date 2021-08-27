#!/usr/bin/env python

from __future__ import print_function
import numpy as np

import time
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, \
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, enable_gravity, set_body_color, \
    get_velocity

from utils.pybullet_tools.body_utils import draw_frame

from copy import copy


class PlanningScenario1(object):
    def __init__(self, enable_drops=False):
        self.num_particle = 160
        self.name_particle = 'drop'
        self.enable_drops = enable_drops
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                set_pose(self.arm_left, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)
                set_pose(self.arm_base, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    'cabinet_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                        fixed_base=True),
                    'drawer_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                        fixed_base=True),
                    'pegboard': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                        fixed_base=True),
                    'region_shelf': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region_table': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region_drawer': load_pybullet("../scenario_description/region_small.urdf", fixed_base=True),
                    'camera1': load_pybullet("../scenario_description/realsense.urdf", fixed_base=True),
                    'cup': load_pybullet("../scenario_description/square_cup.urdf", fixed_base=False),
                    'bowl': load_pybullet("../scenario_description/square_cup_big.urdf", fixed_base=True),
                    'box_S1': load_pybullet("../scenario_description/boxX.urdf", fixed_base=True),
                    'box_S2': load_pybullet("../scenario_description/boxX.urdf", fixed_base=True),

                }

                if self.enable_drops:
                    for i in range(self.num_particle):
                        self.bd_body[self.name_particle + str(i)] = load_pybullet(
                            "../scenario_description/particle.urdf",
                            fixed_base=False)

                self.all_bodies = [b for b in self.bd_body.values()]
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                self.id_to_body = {}

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region_shelf'],
                         Pose(Point(x=-0.38, y=0.85, z=0.774)))

                set_pose(self.bd_body['region_table'],
                         Pose(Point(x=0.45, y=0.75, z=stable_z(self.bd_body['region_table'], self.bd_body['floor']))))
                set_pose(self.bd_body['region_drawer'], Pose(Point(x=-0.14, y=0.90, z=0.001)))
                set_pose(self.bd_body['camera1'],
                         Pose(Point(x=0.20, y=1.3, z=0.9), Euler(0, 145 * np.pi / 180, -100 * np.pi / 180)))

            # set_pose(box1, Pose(Point(x=0.25, y=0.80, z=stable_z(box1, region2))))

        self.movable_bodies = [self.bd_body['cup']]

        self.regions = [self.bd_body['region_shelf'], self.bd_body['region_drawer'], self.bd_body['region_table']]

        self.env_bodies = [b for b in self.all_bodies if (b not in self.regions) and (b not in self.movable_bodies)]

        self.sensors = [self.bd_body['camera1']]

        self.robots = [self.arm_left]

        self.dic_body_info = {}
        for b in self.movable_bodies:
            obj_center, obj_extent = get_center_extent(b)
            body_pose = get_pose(b)
            body_frame = tform_from_pose(body_pose)
            bottom_center = copy(obj_center)
            bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
            bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
            relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
            center_frame = tform_from_pose((obj_center, body_pose[1]))
            relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

            self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

        """If the body should be attached to the region when it is placed."""
        self.dic_body_info[self.bd_body['region_shelf']] = (False,)
        self.dic_body_info[self.bd_body['region_table']] = (False,)
        self.dic_body_info[self.bd_body['region_drawer']] = (True,)

        self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                # initial_jts = np.array([-0.5, 1.4, 1.5, 1.3, 0, 0, 0])
                # initial_jts = np.array([0, 0, 0, 0, 0, 0, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])

                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                cup_x = -0.02
                cup_y = 0.9
                set_pose(self.bd_body['cup'],
                         Pose(Point(x=cup_x, y=cup_y, z=0.0025), Euler(roll=0 * np.pi / 180, yaw=0 * np.pi / 180)))

                set_pose(self.bd_body['bowl'],
                         Pose(Point(x=0.7, y=cup_y, z=0.0025), Euler(roll=0 * np.pi / 180, yaw=0 * np.pi / 180)))

                if self.enable_drops:
                    for i in range(self.num_particle):
                        set_pose(self.bd_body[self.name_particle + str(i)],
                                 Pose(Point(x=np.random.uniform(cup_x - 0.03, cup_x + 0.03, 1),
                                            y=np.random.uniform(cup_y - 0.03, cup_y + 0.03, 1),
                                            z=np.random.uniform(0.08 - 0.04, 0.08 + 0.04, 1))))

                set_pose(self.bd_body['box_S1'], Pose(Point(x=-0.02, y=0.72, z=0.1)))
                set_pose(self.bd_body['box_S2'], Pose(Point(x=-0.02, y=1.08, z=0.1)))

                set_camera(160, -35, 1.8, Point())

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

    def show_view_cone(self, lifetime=10):
        r = Register(self.bd_body['camera1'], self.bd_body['cup'])
        attach_viewcone(self.bd_body['camera1'], depth=1.5)
        draw_frame(tform_from_pose(get_pose(self.bd_body['camera1'])), None)

        r.show()

    def get_rest_drops(self, container):
        if container == 'cup':
            lx = 0.08
            ly = 0.08
            lz = 0.18
        elif container == 'bowl':
            lx = 0.12
            ly = 0.12
            lz = 0.18
        else:
            raise Exception

        assert self.enable_drops
        cup = self.bd_body[container]
        cup_frame = tform_from_pose(get_pose(cup))

        inv_cup_frame = np.linalg.inv(cup_frame)
        ppositions_in_cup = []
        rest_drops = []
        for i in range(self.num_particle):
            p = self.bd_body[self.name_particle + str(i)]
            position, _ = get_pose(p)
            pposition = np.array([position[0], position[1], position[2], 1])
            pposition_in_cup = np.dot(inv_cup_frame, pposition)
            ppositions_in_cup.append(pposition_in_cup)
            if -lx / 2 < pposition_in_cup[0] < lx / 2 and -ly / 2 < \
                    pposition_in_cup[1] < ly / 2 and 0 < pposition_in_cup[2] < \
                    lz:
                rest_drops.append(p)
                set_body_color(p)

        return rest_drops

    def get_drops_info(self, container='cup'):
        if not self.enable_drops:
            return None, None
        rest_drops = self.get_rest_drops(container)
        rest_percentage = float(len(rest_drops)) / self.num_particle
        v = 0
        if len(rest_drops) > 0:
            for p in rest_drops:
                linear, _ = get_velocity(p)
                v += np.linalg.norm(linear)
            v = v / len(rest_drops)
        is_still = v < 0.05
        return rest_percentage, is_still


#######################################################

def get_scn(idx=0):
    list_scn = [PlanningScenario1]
    return list_scn[idx]


def display_scenario():
    connect(use_gui=True)

    scn = get_scn(0)(enable_drops=1)
    scn.reset()

    # scn.show_view_cone()

    enable_gravity()
    for i in range(10000):
        step_simulation()
        time.sleep(0.01)
        if i % 100 == 0:
            rest_percentage, is_still = scn.get_drops_info('bowl')
            print(rest_percentage, is_still)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
