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

for base_x in [0.3, 0.5, 0.7, 0.9]:
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=False,
        initial_base_q=(base_x, 0.0, 0.0),
    )

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

    from pybullet_tools.pr2_utils import set_group_conf
    set_group_conf(env.robot.body, "torso", [0.25])
    set_group_conf(env.robot.body, "head", [0.0, 0.5])
    for _ in range(60):
        p.stepSimulation()

    obs = env.get_observations(wrist_only=True)["head"]
    img = (obs["rgb"] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(f"/home/jx/kitchen/RoboEXP/tmp_close_base{base_x}.png")
    
    cam = env.robot.cameras[0]
    cam_img = cam.get_image(segment=True, segment_links=False)
    seg = cam_img.segmentationMaskBuffer
    cup_mask = (seg == cup.body)
    print(f"base_x={base_x}: cup_pixels={cup_mask.sum()}, unique_seg={np.unique(seg)}")
    
    env.close()
