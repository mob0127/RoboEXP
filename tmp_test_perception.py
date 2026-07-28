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

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]
GROUNDING_DICT = " . ".join(OBJECT_LEVEL_LABELS) + " ."

env = PyBulletExplorationEnv(
    scene_builder="sample_kitchen_mini_scene",
    robot_name="pr2",
    use_gui=False,
    segment=False,
    initial_base_q=(1.5, 0.0, 0.0),
)

print("Loading perception models...")
percept = RoboPercept(grounding_dict=GROUNDING_DICT, lazy_loading=False, device="cuda")
print("Models loaded.")

obs = env.get_observations(wrist_only=True)["head"]
print("Observations captured.")

attrs = percept.get_attributes_from_observations({"head": obs})["head"]
print("Detection results:")
print("  boxes:", len(attrs["pred_boxes"]) if attrs["pred_boxes"] is not None else 0)
print("  phrases:", attrs["pred_phrases"])
print("  masks:", len(attrs["pred_masks"]) if attrs["pred_masks"] is not None else 0)

env.close()
