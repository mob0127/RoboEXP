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
from roboexp.perception.robo_percept import RoboPercept
from roboexp.perception.pybullet_fallback import PyBulletFallbackDetector
from roboexp.memory.robo_memory import RoboMemory

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]
GROUNDING_DICT = " . ".join(OBJECT_LEVEL_LABELS) + " ."

PYBULLET_FALLBACK_CONFIG = {
    "cup": {
        "lower": [0.25, 0.00, 0.00],
        "upper": [1.00, 0.20, 0.15],
        "morph_kernel": 3,
    },
}

LOWER_BOUND = [-2, -2, -0.5]
HIGHER_BOUND = [6, 6, 3]
VOXEL_SIZE = 0.02

env = PyBulletExplorationEnv(
    scene_builder="sample_kitchen_mini_scene",
    robot_name="pr2",
    use_gui=False,
    segment=False,
    initial_base_q=(1.5, 0.0, 0.0),
)

# Add a red test cup on the counter like the TAMP demo.
from pybullet_tools.utils import get_aabb, set_pose, Pose, Point, create_cylinder, RED, set_mass
from world_builder.entities import Object
CYLINDER_RADIUS = 0.06
CYLINDER_HEIGHT = 0.12
counter = env.world.name_to_object("counter")
counter_top = get_aabb(counter.body)[1][2]
z = counter_top + CYLINDER_HEIGHT / 2 + 0.01
body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
cup = env.world.add_object(Object(body, category="cup", name="test_cup"))
set_pose(cup.body, Pose(point=Point(0.65, 0.25, z)))
set_mass(cup.body, 0.1)
import pybullet as p
for _ in range(50):
    p.stepSimulation()

# Add target table.
from pybullet_tools.utils import create_box, BLUE
TABLE_SIZE = (0.50, 0.50, 0.75)
table_body = create_box(*TABLE_SIZE, color=BLUE)
table_z = TABLE_SIZE[2]
set_pose(table_body, Pose(point=Point(1.50, 1.50, table_z)))
table_obj = env.world.add_object(Object(table_body, category="table", name="target_table"))

print("Loading perception models...")
percept = RoboPercept(grounding_dict=GROUNDING_DICT, lazy_loading=False, device="cuda")
fallback = PyBulletFallbackDetector(PYBULLET_FALLBACK_CONFIG)
memory = RoboMemory(lower_bound=LOWER_BOUND, higher_bound=HIGHER_BOUND, voxel_size=VOXEL_SIZE)
print("Models loaded.")

# Move robot head to look at counter.
from pybullet_tools.pr2_utils import set_group_conf
set_group_conf(env.robot.body, "head", [0.0, 0.3])
for _ in range(30):
    p.stepSimulation()

observations = env.get_observations(wrist_only=True)
print("Observations captured.")

attrs = percept.get_attributes_from_observations(observations, visualize=False)
print("Before fallback:")
print("  phrases:", attrs["head"]["pred_phrases"])

attrs = fallback.fill_missing_labels(observations, attrs, ["cup"], replace_existing=False)
print("After fallback:")
print("  phrases:", attrs["head"]["pred_phrases"])

memory.update_memory(
    observations, attrs, OBJECT_LEVEL_LABELS,
    update_scene_graph=True, scene_graph_option=None, visualize=False,
)

print("Scene graph nodes:")
for node_id, node in memory.action_scene_graph.object_nodes.items():
    print(f"  {node_id}: {node.node_label} (parent={node.parent.node_id if node.parent else None}, relation={node.parent_relation})")

env.close()
