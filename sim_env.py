import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

#改为FR3对应的任务
def make_sim_env(task_name):
    """
    Environment for simulated FR3 bi-manual manipulation, with joint position control
    Action space:      [arm_qpos (7),              # absolute joint position
                        gripper_position (1),       # normalized gripper position (0: close, 1: open)]

    Observation space: {"qpos": Concat[ arm_qpos (7),          # absolute joint position
                                        gripper_position (1),  # normalized gripper position (0: close, 1: open)
                       "qvel": Concat[ arm_qvel (7),           # absolute joint velocity (rad)
                                        gripper_velocity (1),  # normalized gripper velocity 
                       "images": {"top": (480x640x3), "wrist": (480x640x3)}
    """
    if 'fr3_pick_place' in task_name:
        xml_path = os.path.join(XML_DIR, f'fr3_pick_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickPlaceTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class FR3Task(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        arm_action = action[:7]                        # 7 关节
        normalized_gripper_action = action[7]          # 1 夹爪

        gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_gripper_action)

        full_gripper_action = [gripper_action, -gripper_action]  # 左右手指反向

        env_action = np.concatenate([arm_action, full_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:7]
        gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[7])]  # left_finger
        return np.concatenate([arm_qpos, gripper_qpos])  # 8 维

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel = qvel_raw[:7]
        gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[7])]
        return np.concatenate([arm_qvel, gripper_qvel])  # 8 维

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['wrist'] = physics.render(height=480, width=640, camera_id='wrist')        
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickPlaceTask(FR3Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:9] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[9:] # 9 维 arm+gripper 之后
        return env_state

    def get_reward(self, physics):
        # 单臂：抓取方块并提起
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = ("red_box", "fr3_left_finger") in all_contact_pairs or \
                        ("red_box", "fr3_right_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        # 方块当前位置
        box_pos = physics.named.data.qpos['red_box_joint'][:3]
        target_pos = np.array([0.5, 0.2, 0.1])  # 目标位置
        at_target = np.linalg.norm(box_pos - target_pos) < 0.03

        reward = 0
        if touch_gripper:                    # 触碰方块
            reward = 1
        if touch_gripper and not touch_table: # 抓起
            reward = 2
        if touch_gripper and not touch_table and at_target:
            reward = 3        
        
        return reward


def get_action(master_bot):
    """FR3 单臂遥操作：从主手读取关节角度"""
    action = np.zeros(8)                          # 原来是 14
    action[:7] = master_bot.dxl.joint_states.position[:7]   # 7 关节（原来是 6）
    gripper_pos = master_bot.dxl.joint_states.position[7]
    action[7] = MASTER_GRIPPER_POSITION_NORMALIZE_FN(gripper_pos)
    return action

def test_sim_teleop():
    """FR3 仿真遥操作测试"""
    # 真机时替换为 FR3 主手驱动
    # from fr3_interface import FR3Arm
    # master_bot = FR3Arm(...)
    
    BOX_POSE[0] = [0.5, 0, 0.05, 1, 0, 0, 0]   # 方块在桌面

    env = make_sim_env('fr3_pick_place_scripted')
    ts = env.reset()
    episode = [ts]
    
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['top'])   # 原来是 'angle'
    plt.ion()

    for t in range(400):
        # 暂时用小幅随机动作替代真机遥操作
        action = np.zeros(8)
        action[:7] = ts.observation['qpos'][:7] + np.random.uniform(-0.01, 0.01, 7)
        action[7] = np.random.uniform(0, 1)  # 随机夹爪
        # 真机时: action = get_action(master_bot)
        
        ts = env.step(action)
        episode.append(ts)
        plt_img.set_data(ts.observation['images']['top'])
        plt.pause(0.02)
    plt.close()

if __name__ == '__main__':
    test_sim_teleop()
