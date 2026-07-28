# Xiao : 新增文件，用于本仓库对上游的扩展。
import sys, os, numpy as np, imageio, random
sys.path.insert(0, '/home/jx/kitchen/kitchen-worlds')
sys.path.insert(0, '/home/jx/kitchen/kitchen-roboexp/pybullet_planning')
sys.path.insert(0, '/home/jx/kitchen/RoboEXP')
from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf
from pybullet_tools.utils import get_aabb, get_link_pose, link_from_name, point_from_pose
from experiments.cup_dynamic_demo_pr2 import set_pr2_root, add_test_cup, HOME_CONF, REACH_CONF, PRE_GRASP_CONF, GRASP_CONTACT_CONF
import pybullet as p

random.seed(42)
np.random.seed(42)
env = PyBulletExplorationEnv(scene_builder='sample_kitchen_mini_scene', robot_name='pr2', use_gui=False, segment=False, initial_base_q=(0.0,0.0,0.0))
counter = env.world.name_to_object('counter')
counter_top = get_aabb(counter.body)[1][2]

for bx, cup_x in [(1.3,0.68),(1.4,0.78),(1.5,0.88),(1.6,0.98)]:
    if cup_x >= 1.0: continue
    set_pr2_root(env, bx, 0.0, np.pi)
    set_group_conf(env.robot.body, 'torso', [0.25])
    set_group_conf(env.robot.body, 'head', [0.0,0.3])
    set_arm_conf(env.robot.body, 'left', HOME_CONF)
    for _ in range(20): p.stepSimulation()
    cup = add_test_cup(env, counter_top, [cup_x,0.0], name='test_cup')
    for name, conf in [('HOME',HOME_CONF),('REACH',REACH_CONF),('PRE',PRE_GRASP_CONF),('CONTACT',GRASP_CONTACT_CONF)]:
        set_arm_conf(env.robot.body, 'left', conf)
        for _ in range(20): p.stepSimulation()
        pos = point_from_pose(get_link_pose(env.robot.body, link_from_name(env.robot.body, 'l_gripper_tool_frame')))
    img = env.get_global_image(width=640, height=480)
    fname = f'/home/jx/kitchen/RoboEXP/clip_bx{bx}_cx{cup_x}.png'
    imageio.imwrite(fname, img)
    print('saved', fname)
