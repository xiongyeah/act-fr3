"""获取 FR3 在 home 姿态下末端的真实位姿，用于初始化 mocap"""
import numpy as np
import sys
sys.path.insert(0, '.')

from dm_control.mujoco.wrapper import MjModel
from dm_control import mujoco
from constants import XML_DIR, START_ARM_POSE

xml_path = XML_DIR + 'fr3_ee_pick_cube.xml'
model = MjModel.from_xml_path(xml_path)
physics = mujoco.Physics.from_model(model)

with physics.reset_context():
    physics.named.data.qpos[:9] = START_ARM_POSE

mujoco.mj_forward(model._model, physics.data._data)

pos = physics.named.data.xpos['fr3_link7'].copy()
quat = physics.named.data.xquat['fr3_link7'].copy()

print(f"EE pos:  [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
print(f"EE quat: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
