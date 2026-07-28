# Xiao : 新增文件，用于本仓库对上游的扩展。
import sys, os, numpy as np, imageio, random
sys.path.insert(0, '/home/jx/kitchen/kitchen-worlds')
sys.path.insert(0, '/home/jx/kitchen/kitchen-roboexp/pybullet_planning')
sys.path.insert(0, '/home/jx/kitchen/RoboEXP')
from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf, get_arm_joints
from pybullet_tools.utils import get_aabb, get_link_pose, link_from_name, point_from_pose, set_joint_positions
from pybullet_tools.pr2_utils import SIDE_HOLDING_LEFT_ARM
from experiments.cup_dynamic_demo_pr2 import (
    set_pr2_root, add_test_cup, REACH_CONF, PRE_GRASP_CONF,
    GRASP_CONTACT_CONF, LIFT_CONF
)
import pybullet as p

random.seed(42)
np.random.seed(42)
env = PyBulletExplorationEnv(scene_builder='sample_kitchen_mini_scene', robot_name='pr2', use_gui=False, segment=False, initial_base_q=(0.0,0.0,0.0))
counter = env.world.name_to_object('counter')
counter_top = get_aabb(counter.body)[1][2]
base = [1.5, 0.0, np.pi]
set_pr2_root(env, *base)
set_group_conf(env.robot.body, 'torso', [0.25])
set_group_conf(env.robot.body, 'head', [0.0,0.3])
set_arm_conf(env.robot.body, 'left', SIDE_HOLDING_LEFT_ARM)
for _ in range(20): p.stepSimulation()
cup = add_test_cup(env, counter_top, [0.848,0.0], name='test_cup')

def collisions():
    return len(p.getContactPoints(env.robot.body, counter.body))

for label, conf in [('HOME',SIDE_HOLDING_LEFT_ARM),('REACH',REACH_CONF),('PRE',PRE_GRASP_CONF),('CONTACT',GRASP_CONTACT_CONF),('LIFT',LIFT_CONF)]:
    # move smoothly
    arm_joints = get_arm_joints(env.robot.body, 'left')
    cur = [p.getJointState(env.robot.body, j)[0] for j in arm_joints]
    target = conf
    for i in range(1, 13):
        alpha = i/12
        interp = [cur[j]*(1-alpha)+target[j]*alpha for j in range(7)]
        set_joint_positions(env.robot.body, arm_joints, interp)
        for _ in range(5): p.stepSimulation()
    pos = point_from_pose(get_link_pose(env.robot.body, link_from_name(env.robot.body, 'l_gripper_tool_frame')))
    col = collisions()
    print(label, 'tool', [round(x,3) for x in pos], 'collisions', col)
    img = env.get_global_image(width=640, height=480)
    imageio.imwrite(f'/home/jx/kitchen/RoboEXP/seq2_{label}.png', img)
