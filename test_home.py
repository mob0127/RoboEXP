# Xiao : 新增文件，用于本仓库对上游的扩展。
import sys, os, numpy as np, imageio, random
sys.path.insert(0, '/home/jx/kitchen/kitchen-worlds')
sys.path.insert(0, '/home/jx/kitchen/kitchen-roboexp/pybullet_planning')
sys.path.insert(0, '/home/jx/kitchen/RoboEXP')
from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf
from pybullet_tools.utils import get_aabb, get_link_pose, link_from_name, point_from_pose
from experiments.cup_dynamic_demo_pr2 import set_pr2_root
from pybullet_tools.pr2_utils import REST_LEFT_ARM, TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, WIDE_LEFT_ARM, CENTER_LEFT_ARM
import pybullet as p

random.seed(42)
np.random.seed(42)
env = PyBulletExplorationEnv(scene_builder='sample_kitchen_mini_scene', robot_name='pr2', use_gui=False, segment=False, initial_base_q=(0.0,0.0,0.0))
counter = env.world.name_to_object('counter')
counter_top = get_aabb(counter.body)[1][2]
set_pr2_root(env, 1.5, 0.0, np.pi)
set_group_conf(env.robot.body, 'torso', [0.25])
set_group_conf(env.robot.body, 'head', [0.0,0.3])
for _ in range(20): p.stepSimulation()
for name, conf in [('TOP',TOP_HOLDING_LEFT_ARM),('SIDE',SIDE_HOLDING_LEFT_ARM),('WIDE',WIDE_LEFT_ARM),('CENTER',CENTER_LEFT_ARM)]:
    set_arm_conf(env.robot.body, 'left', conf)
    for _ in range(20): p.stepSimulation()
    pos = point_from_pose(get_link_pose(env.robot.body, link_from_name(env.robot.body, 'l_gripper_tool_frame')))
    col = len(p.getContactPoints(env.robot.body, counter.body))
    print(name, 'tool', [round(x,3) for x in pos], 'collisions', col)
    img = env.get_global_image(width=640, height=480)
    imageio.imwrite(f'/home/jx/kitchen/RoboEXP/home_{name}.png', img)
