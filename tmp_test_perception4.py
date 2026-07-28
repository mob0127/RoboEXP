# Xiao : 新增文件，用于本仓库对上游的扩展。
import sys
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

import random
import numpy as np
from pybullet_tools.utils import set_random_seed
random.seed(0)
np.random.seed(0)
set_random_seed(0)

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.utils import get_aabb, set_pose, Pose, Point, create_cylinder, RED, set_mass
from world_builder.entities import Object
import pybullet as p
from PIL import Image

env = PyBulletExplorationEnv(
    scene_builder="sample_kitchen_mini_scene",
    robot_name="pr2",
    use_gui=False,
    segment=False,
    initial_base_q=(1.5, 0.0, 0.0),
)

# Add a red test cup on the counter like the TAMP demo.
CYLINDER_RADIUS = 0.06
CYLINDER_HEIGHT = 0.12
counter = env.world.name_to_object("counter")
counter_top = get_aabb(counter.body)[1][2]
z = counter_top + CYLINDER_HEIGHT / 2 + 0.01
body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
cup = env.world.add_object(Object(body, category="cup", name="test_cup"))
set_pose(cup.body, Pose(point=Point(0.65, 0.25, z)))
set_mass(cup.body, 0.1)
for _ in range(50):
    p.stepSimulation()

# Set base to face counter like TAMP demo.
from roboexp.env import pr2_skills
pr2_skills.set_pr2_root(env, 1.5, 0.0, np.pi)

# Move robot head to look at counter.
from pybullet_tools.pr2_utils import set_group_conf
set_group_conf(env.robot.body, "torso", [0.25])
set_group_conf(env.robot.body, "head", [0.0, 0.3])
for _ in range(30):
    p.stepSimulation()

observations = env.get_observations(wrist_only=True)
obs = observations["head"]
img = (obs["rgb"] * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img).save("/home/jx/kitchen/RoboEXP/tmp_head_view2.png")
print("Saved head view to tmp_head_view2.png")

# Check where red pixels are.
red = obs["rgb"][:, :, 0]
green = obs["rgb"][:, :, 1]
blue = obs["rgb"][:, :, 2]
mask = (red > 0.25) & (red > green + 0.15) & (red > blue + 0.15)
print("Red pixel count:", mask.sum())
if mask.sum() > 0:
    ys, xs = np.where(mask)
    print("Red bbox:", xs.min(), ys.min(), xs.max(), ys.max())

env.close()
