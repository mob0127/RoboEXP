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
set_pr2_root(env, 1.2, 0.0, np.pi)
set_group_conf(env.robot.body, 'torso', [0.25])
set_group_conf(env.robot.body, 'head', [0.0,0.3])
set_arm_conf(env.robot.body, 'left', [0.0]*7)
for _ in range(20): p.stepSimulation()
cup = add_test_cup(env, counter_top, [0.58,0.0], name='test_cup')

# find a high tool pose above cup
best = None
for i in range(200):
    yaw = random.uniform(-np.pi, np.pi)
    pitch = random.uniform(-np.pi/2, 0)  # look down
    roll = random.uniform(-np.pi, np.pi)
    tool_pose = Pose(point=Point(0.58,0.0,1.85), euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
    conf = sample_tool_ik(env.robot.body, 'left', tool_pose, torso_limits=(0.25,0.25), max_attempts=5)
    if conf is None:
        continue
    joints = get_torso_arm_joints(env.robot.body, 'left')
    set_joint_positions(env.robot.body, joints, conf)
    for _ in range(10): p.stepSimulation()
    actual = point_from_pose(get_link_pose(env.robot.body, link_from_name(env.robot.body, 'l_gripper_tool_frame')))
    err = np.linalg.norm(np.array(actual)-np.array([0.58,0.0,1.85]))
    if err > 0.05: continue
    # check collisions between arm and counter
    counter_body = counter.body
    robot = env.robot.body
    # get arm links: children of arm joints? use all links except base
    num_joints = p.getNumJoints(robot)
    collisions = 0
    for j in range(num_joints):
        # skip base and head etc? just check all vs counter
        if p.getContactPoints(robot, counter_body, j, -1):
            collisions += 1
    if best is None or collisions < best[0]:
        best = (collisions, conf, actual)
    if collisions == 0:
        print('NO COLLISION high reach conf', [round(x,3) for x in conf[1:]])
        break
print('best collisions', best[0] if best else None)
