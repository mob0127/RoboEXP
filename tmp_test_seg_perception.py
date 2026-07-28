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
from roboexp.memory.robo_memory import RoboMemory

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]
GROUNDING_DICT = " . ".join(OBJECT_LEVEL_LABELS) + " ."
LOWER_BOUND = [-2, -2, -0.5]
HIGHER_BOUND = [6, 6, 3]
VOXEL_SIZE = 0.02

env = PyBulletExplorationEnv(
    scene_builder="sample_kitchen_mini_scene",
    robot_name="pr2",
    use_gui=False,
    segment=False,
    initial_base_q=(0.5, 0.0, 0.0),
)

# Add cup.
from pybullet_tools.utils import get_aabb, set_pose, Pose, Point, create_cylinder, RED, set_mass
from world_builder.entities import Object
import pybullet as p
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

# Add target table.
from pybullet_tools.utils import create_box, BLUE
TABLE_SIZE = (0.50, 0.50, 0.75)
table_body = create_box(*TABLE_SIZE, color=BLUE)
table_z = TABLE_SIZE[2]
set_pose(table_body, Pose(point=Point(1.50, 1.50, table_z)))
table_obj = env.world.add_object(Object(table_body, category="table", name="target_table"))

from pybullet_tools.pr2_utils import set_group_conf
set_group_conf(env.robot.body, "torso", [0.25])
set_group_conf(env.robot.body, "head", [0.0, 0.3])
for _ in range(60):
    p.stepSimulation()

print("Loading perception models...")
percept = RoboPercept(grounding_dict=GROUNDING_DICT, lazy_loading=False, device="cuda")
memory = RoboMemory(lower_bound=LOWER_BOUND, higher_bound=HIGHER_BOUND, voxel_size=VOXEL_SIZE)
print("Models loaded.")

# Get observations with segmentation.
observations = env.get_observations(wrist_only=True)
obs = observations["head"]

# Also get segmentation image.
cam = env.robot.cameras[0]
cam_img = cam.get_image(segment=True, segment_links=False)
seg = cam_img.segmentationMaskBuffer
print("Segmentation shape:", seg.shape)
print("Unique seg values:", np.unique(seg))
print("Cup body:", cup.body)
cup_mask = (seg == cup.body)
print("Cup pixels in segmentation:", cup_mask.sum())

# Run normal perception.
attrs = percept.get_attributes_from_observations(observations, visualize=False)
print("Before injection:", attrs["head"]["pred_phrases"])

# Inject cup detection from segmentation.
MASK_FEAT_DIM = 512
if cup_mask.sum() > 0:
    ys, xs = np.where(cup_mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    mask = cup_mask[None, ...].astype(bool)
    feat = np.ones((1, MASK_FEAT_DIM), dtype=np.float32) * 1e-3
    
    existing_boxes = []
    existing_masks = []
    existing_phrases = []
    existing_feats = []
    if len(attrs["head"]["pred_boxes"]) > 0:
        for j, phrase in enumerate(attrs["head"]["pred_phrases"]):
            existing_boxes.append(attrs["head"]["pred_boxes"][j][None, :])
            existing_masks.append(attrs["head"]["pred_masks"][j][None, ...].astype(bool))
            existing_phrases.append(phrase)
            existing_feats.append(attrs["head"]["mask_feats"][j][None, :])
    
    existing_boxes.append(box)
    existing_masks.append(mask)
    existing_phrases.append("cup:1.0")
    existing_feats.append(feat)
    
    attrs["head"]["pred_boxes"] = np.concatenate(existing_boxes, axis=0)
    attrs["head"]["pred_masks"] = np.concatenate(existing_masks, axis=0)
    attrs["head"]["pred_phrases"] = existing_phrases
    attrs["head"]["mask_feats"] = np.concatenate(existing_feats, axis=0)

print("After injection:", attrs["head"]["pred_phrases"])

memory.update_memory(
    observations, attrs, OBJECT_LEVEL_LABELS,
    filter_masks={},
    update_scene_graph=True, scene_graph_option=None, visualize=False,
)

print("Scene graph nodes:")
for node_id, node in memory.action_scene_graph.object_nodes.items():
    print(f"  {node_id}: {node.node_label} (parent={node.parent.node_id if node.parent else None}, relation={node.parent_relation})")

env.close()
