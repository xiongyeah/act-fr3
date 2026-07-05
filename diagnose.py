"""诊断 FR3 抓取问题"""
import numpy as np
import sys
sys.path.insert(0, '.')
from dm_control.mujoco.wrapper import MjModel
from dm_control import mujoco
from constants import XML_DIR, START_ARM_POSE
from utils import sample_box_pose
from ee_sim_env import make_ee_sim_env
from scripted_policy import PickAndPlacePolicy

# 1. 检查 mocap 初始化是否对齐
print("=" * 50)
print("1. 检查 EE 初始化")
env = make_ee_sim_env('fr3_pick_place_scripted')
ts = env.reset()
physics = env._physics

ee_pos = physics.named.data.xpos['fr3_link7'].copy()
mocap_pos = physics.data.mocap_pos[0].copy()
dist = np.linalg.norm(ee_pos - mocap_pos)
print(f"  fr3_link7 位置: {ee_pos}")
print(f"  mocap 位置:     {mocap_pos}")
print(f"  距离差: {dist:.4f} {'⚠️ 未对齐' if dist > 0.01 else '✓ 对齐'}")

# 2. 检查方块位置
print("\n2. 检查方块位置")
box_pos = physics.named.data.qpos['red_box_joint'][:3].copy()
print(f"  方块位置: {box_pos}")
print(f"  末端到方块距离: {np.linalg.norm(ee_pos - box_pos):.3f}")

# 3. 检查 action 维度
print("\n3. 检查 action")
policy = PickAndPlacePolicy(inject_noise=False)
action = policy(ts)
print(f"  action shape: {action.shape} (期望 8)")
print(f"  action: xyz={action[:3]}, quat={action[3:7]}, gripper={action[7]:.3f}")

# 4. 跑几步看 reward
print("\n4. 跑几步看 reward")
for step in range(5):
    ts = env.step(action)
    action = policy(ts)
    print(f"  step {step}: reward={ts.reward}, gripper_qpos={ts.observation['qpos'][7]:.3f}")

# 5. 检查夹爪碰撞几何
print("\n5. 检查夹爪 geom")
for i in range(physics.model.ngeom):
    name = physics.model.id2name(i, 'geom')
    if 'finger' in (name or ''):
        print(f"  {name}: contype={physics.model.geom_contype[i]}, conaffinity={physics.model.geom_conaffinity[i]}")

print("\n诊断完成")
