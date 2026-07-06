import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'fr3_pick_place' in task_name:
        xml_path = os.path.join(XML_DIR, f'fr3_ee_scene.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickPlaceEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class FR3EETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        # Joint limits from fr3.xml
        self.q_min = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065,  0.5445, -3.0159])
        self.q_max = np.array([ 2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159])

    @staticmethod
    def _quat_mul(q1, q2):
        """Quaternion multiplication (w,x,y,z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _solve_ik(self, physics, target_pos, target_quat, max_iter=30, tol=1e-4, damping=0.05, alpha=0.8):
        """Numerical IK using damped least squares Jacobian pseudo-inverse."""
        q = physics.data.qpos[:7].copy()
        eps = 1e-5

        for _ in range(max_iter):
            # Forward kinematics
            physics.data.qpos[:7] = q
            physics.forward()
            pos = physics.named.data.xpos['fr3_link7'].copy()
            quat = physics.named.data.xquat['fr3_link7'].copy()

            # Position error
            pos_err = target_pos - pos

            # Orientation error via quaternion difference
            q_cur_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
            q_err = self._quat_mul(target_quat, q_cur_inv)
            if q_err[0] < 0:
                q_err = -q_err
            ori_err = 2.0 * q_err[1:]  # small-angle approximation for angular velocity

            err = np.concatenate([pos_err, ori_err])
            if np.linalg.norm(err) < tol:
                break

            # Numerical Jacobian (6×7)
            J = np.zeros((6, 7))
            for i in range(7):
                q_pert = q.copy()
                q_pert[i] += eps
                physics.data.qpos[:7] = q_pert
                physics.forward()
                pos_p = physics.named.data.xpos['fr3_link7'].copy()
                quat_p = physics.named.data.xquat['fr3_link7'].copy()

                J[:3, i] = (pos_p - pos) / eps

                q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
                dq = self._quat_mul(quat_p, q_inv)
                if dq[0] < 0:
                    dq = -dq
                J[3:, i] = 2.0 * dq[1:] / eps

            # Damped pseudo-inverse: dq = J^T (JJ^T + λI)^{-1} err
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damping * np.eye(6), err)
            q = q + alpha * dq
            q = np.clip(q, self.q_min, self.q_max)

        # Restore original qpos (IK only reads, shouldn't permanently change state)
        return q

    def before_step(self, action, physics):
        # action: [x, y, z, qw, qx, qy, qz, gripper]  8 维
        target_pos = action[:3]
        target_quat = action[3:7]

        # IK: map end-effector target → joint angles
        # 保存当前 qpos，IK 在迭代中会污染它
        qpos_backup = physics.data.qpos[:7].copy()
        target_q = self._solve_ik(physics, target_pos, target_quat)
        # 恢复 qpos，让位置控制器有误差可追 = 手臂有刚度
        physics.data.qpos[:7] = qpos_backup
        physics.forward()
        np.copyto(physics.data.ctrl[:7], target_q)

        # Gripper
        g_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action[7])
        np.copyto(physics.data.ctrl[7:9], [g_ctrl, -g_ctrl])

        # Debug prints
        # step = int(round(physics.data.time / 0.02))
        # if step in (0, 399):
        #     qpos_backup = physics.data.qpos[:7].copy()
        #     physics.data.qpos[:7] = target_q
        #     physics.forward()
        #     box_pos = physics.named.data.xpos['box'].copy()
        #     # Compute local Z direction (finger direction)
        #     from pyquaternion import Quaternion
        #     physics.data.qpos[:7] = qpos_backup
        #     physics.forward()
        #     print(f"t={step}")
        #     print(f"  box={np.round(box_pos, 3)}")

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:9] = START_ARM_POSE

        # Forward to get link7 pose from START_ARM_POSE
        physics.forward()
        link7_pos = physics.named.data.xpos['fr3_link7'].copy()
        link7_quat = physics.named.data.xquat['fr3_link7'].copy()
        np.copyto(physics.data.mocap_pos[0], link7_pos)
        np.copyto(physics.data.mocap_quat[0], link7_quat)

        # Joint controllers start at START_ARM_POSE
        np.copyto(physics.data.ctrl[:7], START_ARM_POSE[:7])
        np.copyto(physics.data.ctrl[7:9], [
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:7]
        gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[7])]
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
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['images']['wrist'] = physics.render(height=480, width=640, camera_id='wrist')
        obs['mocap_pose'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class PickPlaceEETask(FR3EETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[9:]
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
        target_pos = np.array([0.7, 0.2, 0.02])  # 目标位置
        at_target = np.linalg.norm(box_pos - target_pos) < 0.04
        # print(f"at_target={np.linalg.norm(box_pos - target_pos)}")

        reward = 0
        if touch_gripper:                    # 触碰方块
            reward = 1
        if touch_gripper and not touch_table: # 抓起
            reward = 2
        if at_target:
            reward = 3        
        
        return reward
