# Xiao : 新增文件，用于本仓库对上游的扩展。
import sys, os, numpy as np, random
sys.path.insert(0, '/home/jx/kitchen/kitchen-worlds')
sys.path.insert(0, '/home/jx/kitchen/kitchen-roboexp/pybullet_planning')
sys.path.insert(0, '/home/jx/kitchen/RoboEXP')
from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf, get_torso_arm_joints
from pybullet_tools.utils import get_aabb, get_link_pose, link_from_name, point_from_pose, Pose, Point, Euler, quat_from_euler, set_joint_positions
from pybullet_tools.ikfast.pr2.ik import sample_tool_ik
import pybullet as p

random.seed(42)
np.random.seed(42)
env = PyBulletExplorationEnv(scene_builder='sample_kitchen_mini_scene', robot_name='pr2', use_gui=False, segment=False, initial_base_q=(0.0,0.0,0.0))
from experiments.cup_dynamic_demo_pr2 import set_pr2_root, add_test_cup

counter = env.world.name_to_object('counter')
counter_top = get_aabb(counter.body)[1][2]
base = [1.3, 0.0, np.pi]
set_pr2_root(env, *base)
set_group_conf(env.robot.body, 'torso', [0.25])
set_group_conf(env.robot.body, 'head', [0.0,0.3])
set_arm_conf(env.robot.body, 'left', [0.0]*7)
for _ in range(20): p.stepSimulation()
cup = add_test_cup(env, counter_top, [0.68,0.0], name='test_cup')

counter_body = counter.body
robot = env.robot.body

def arm_collisions():
    pts = p.getContactPoints(robot, counter_body)
    return len(pts)

target = [0.68, 0.0, 1.43]
best = None
for i in range(300):
    roll = random.uniform(-np.pi, np.pi)
    pitch = random.uniform(-np.pi/2, np.pi/2)
    yaw = random.uniform(-np.pi, np.pi)
    tool_pose = Pose(point=Point(*target), euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
    conf = sample_tool_ik(env.robot.body, 'left', tool_pose, torso_limits=(0.25,0.25), max_attempts=5)
    if conf is None:
        continue
    joints = get_torso_arm_joints(env.robot.body, 'left')
    set_joint_positions(env.robot.body, joints, conf)
    for _ in range(10): p.stepSimulation()
    actual = point_from_pose(get_link_pose(env.robot.body, link_from_name(env.robot.body, 'l_gripper_tool_frame')))
    err = np.linalg.norm(np.array(actual)-np.array(target))
    if err > 0.05: continue
    col = arm_collisions()
    if best is None or col < best[0] or (col == best[0] and err < best[1]):
        best = (col, err, conf, (roll,pitch,yaw), actual)
    if col == 0:
        print('NO COLLISION contact conf', [round(x,3) for x in conf[1:]], 'rpy', roll,pitch,yaw)
        break
print('best', best[0] if best else None, best[1] if best else None, best[4] if best else None)
