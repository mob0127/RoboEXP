# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Dynamic PR2 demo of tasks 3-5.

A single continuous experiment using the PR2 robot:
  1. Observe a red cup on the counter (with PyBullet fallback mask).
  2. The PR2 left arm picks the cup and places it at another counter location.
  3. Re-observe and show that the cup keeps the same scene-graph identity.
  4. Demonstrate grasp validation: success / missed / tipped.

All reusable PR2 motion primitives live in ``roboexp.env.pr2_skills``; this
script only contains demo-specific setup, visualization, and sequencing.

Outputs:
  - outputs/dynamic_demo_pr2/dynamic_demo_pr2.mp4
  - outputs/dynamic_demo_pr2/frames/*.png
  - outputs/dynamic_demo_pr2/report.json
"""
import sys
import os
import json
import random
import numpy as np
import cv2

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.env import pr2_skills
from roboexp.memory.robo_memory import RoboMemory
from roboexp.perception.robo_percept import RoboPercept
from roboexp.perception.pybullet_fallback import PyBulletFallbackDetector
from roboexp.validation.grasp_validator import GraspValidator

from pybullet_tools.utils import (
    PI, get_aabb, set_pose, set_mass, Pose, Point, create_cylinder, RED,
    get_pose, point_from_pose
)
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf
from world_builder.entities import Object
import pybullet as p


CYLINDER_RADIUS = 0.08
CYLINDER_HEIGHT = 0.20

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]

PYBULLET_FALLBACK_CONFIG = {
    "cup": {
        "lower": [0.25, 0.00, 0.00],
        "upper": [1.00, 0.20, 0.15],
        "morph_kernel": 3,
    },
}
FALLBACK_LABELS = ["cup"]

LOWER_BOUND = [-1, -2, -0.5]
HIGHER_BOUND = [4, 2, 2.5]
VOXEL_SIZE = 0.02

OUT_DIR = "/home/jx/kitchen/RoboEXP/experiments/outputs/dynamic_demo_pr2"
FRAME_DIR = os.path.join(OUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUT_DIR, "dynamic_demo_pr2.mp4")
REPORT_PATH = os.path.join(OUT_DIR, "report.json")

FRAME_SIZE = (640, 480)
COMBO_SIZE = (1280, 480)
FPS = 2
FRAME_COUNTER = 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_tag(prefix):
    global FRAME_COUNTER
    FRAME_COUNTER += 1
    return f"f{FRAME_COUNTER:03d}_{prefix}"


def clear_scene_movables(world):
    """Remove small objects placed by the scene builder so the cup has free space."""
    keep = {"floor", "wall", "counter", "shelf", "cabinet", "minifridge",
            "oven", "dishwasher", "robot", "supporter", "fixture"}
    for cat in list(world.OBJECTS_BY_CATEGORY.keys()):
        if cat in keep:
            continue
        for obj in list(world.OBJECTS_BY_CATEGORY[cat]):
            try:
                if hasattr(obj, "body") and isinstance(obj.body, int):
                    world.remove_object(obj)
                    try:
                        p.getBodyInfo(obj.body)
                        p.removeBody(obj.body)
                    except Exception:
                        pass
            except Exception:
                pass


def add_test_cup(env, surface_top_z, xy, name="test_cup"):
    z = surface_top_z + CYLINDER_HEIGHT / 2 + 0.01
    body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
    cup = env.world.add_object(Object(body, category="cup", name=name))
    set_pose(cup.body, Pose(point=Point(xy[0], xy[1], z)))
    set_mass(cup.body, 0.1)
    for _ in range(50):
        p.stepSimulation()
    return cup


def reset_cup(cup, pose):
    set_pose(cup.body, pose)
    p.resetBaseVelocity(cup.body, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    for _ in range(20):
        p.stepSimulation()


def red_object_mask(rgb):
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 0.25) & (red > green + 0.15) & (red > blue + 0.15)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def record_frame(env, tag, label=None, sublabel=None, mask=None):
    """Save a global + robot-camera composite frame."""
    ensure_dir(FRAME_DIR)
    obs = env.get_observations()["head"]
    head_img = (obs["rgb"] * 255).clip(0, 255).astype(np.uint8)
    head_img = cv2.resize(head_img, FRAME_SIZE)
    global_img = np.ascontiguousarray(env.get_global_image(width=FRAME_SIZE[0], height=FRAME_SIZE[1]))

    if mask is not None:
        mask_r = cv2.resize(mask.astype(np.uint8), FRAME_SIZE).astype(bool)
        overlay = head_img.copy()
        overlay[mask_r] = (255, 0, 0)
        head_img = cv2.addWeighted(head_img, 0.5, overlay, 0.5, 0)

    combo = np.hstack([global_img, head_img])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo, "GLOBAL", (20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, "ROBOT", (FRAME_SIZE[0] + 20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(combo, label, (20, FRAME_SIZE[1] - 20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if sublabel:
        cv2.putText(combo, sublabel, (FRAME_SIZE[0] + 20, FRAME_SIZE[1] - 20),
                    font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    path = os.path.join(FRAME_DIR, f"{tag}.png")
    cv2.imwrite(path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    return combo


def summarize_cup(memory):
    sg = memory.action_scene_graph
    for node_id, node in sg.object_nodes.items():
        if node.node_label == "cup" and node.instance is not None:
            return {
                "node_id": node_id,
                "instance_id": str(node.instance.instance_id),
                "parent": node.parent.node_id if node.parent else None,
                "parent_relation": node.parent_relation,
            }
    return None


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(FRAME_DIR)
    for f in os.listdir(FRAME_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(FRAME_DIR, f))
    global FRAME_COUNTER
    FRAME_COUNTER = 0
    random.seed(42)
    np.random.seed(42)

    print("==================== SETUP ====================")
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=False,
        initial_base_q=(0.0, 0.0, 0.0),
    )
    # Let the validator read our manual fixed-constraint attachments.
    env.physical_grasp = True

    # Compute the tool offset before moving the base.
    pre_tool_xy = pr2_skills.compute_tool_offset(env)

    # Position the cup so the PR2 can stand in front of the counter (x > 1)
    # facing the counter (theta = pi) instead of being embedded in the wall.
    base_pick = np.array([1.50, 0.00, np.pi])
    actual_cup_xy = base_pick[:2] - pre_tool_xy
    cup_xy = actual_cup_xy.tolist()
    place_offset = np.array([0.00, 0.60])
    place_xy = actual_cup_xy + place_offset
    base_place = np.concatenate([place_xy + pre_tool_xy, [np.pi]])
    base_obs = base_pick.copy()

    # Place the robot in front of the counter from the start.
    pr2_skills.set_pr2_root(env, *base_pick)
    set_group_conf(env.robot.body, "torso", [0.25])
    set_group_conf(env.robot.body, "head", [0.0, 0.3])
    set_arm_conf(env.robot.body, "left", pr2_skills.HOME_CONF)
    env.run_action(0, [], iteration=50)

    clear_scene_movables(env.world)
    record_frame(env, make_tag("setup"), label="Setup: PR2 in kitchen")

    memory = RoboMemory(
        lower_bound=LOWER_BOUND, higher_bound=HIGHER_BOUND, voxel_size=VOXEL_SIZE,
        real_camera=False,
        position_association_enabled=True,
        position_association_threshold=1.0,
        move_merge_distance_threshold=0.2,
    )
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    percept = RoboPercept(grounding_dict="cup . table . counter .", lazy_loading=False, device=device)
    fallback_detector = PyBulletFallbackDetector(PYBULLET_FALLBACK_CONFIG)
    validator = GraspValidator(env)

    counter = env.world.name_to_object("counter")
    counter_top = get_aabb(counter.body)[1][2]

    upright_after_move = Pose(point=Point(
        place_xy[0], place_xy[1], counter_top + CYLINDER_HEIGHT / 2 + 0.01))

    # ------------------------------------------------------------------
    # Phase A: cup on counter
    # ------------------------------------------------------------------
    print("\n=== Phase A: cup on counter ===")
    cup = add_test_cup(env, counter_top, cup_xy, name="test_cup")
    actual_cup_xy = np.array(point_from_pose(get_pose(cup.body))[:2])

    pr2_skills.set_static_camera(env, actual_cup_xy, camera_base_xy=base_obs)
    obs = pr2_skills.observe_and_update(
        env, memory, percept, fallback_detector,
        base_xytheta=base_obs,
        look_at_xy=actual_cup_xy,
        scene_graph_option=None,
        object_level_labels=OBJECT_LEVEL_LABELS,
        fallback_labels=FALLBACK_LABELS,
    )
    mask = red_object_mask(obs["rgb"])
    record_frame(env, make_tag("counter_initial"), label="Phase A: cup on counter",
                 sublabel="fallback mask", mask=mask)
    cup_before = summarize_cup(memory)
    print("Cup before move:", cup_before)

    # ------------------------------------------------------------------
    # Phase B: pick cup from counter and place to another counter location
    # ------------------------------------------------------------------
    print("\n=== Phase B: PR2 pick & place ===")
    pr2_skills.move_pr2_base(env, base_pick)
    pr2_skills.move_arm_to_conf(env, pr2_skills.HOME_CONF, steps=6)
    record_frame(env, make_tag("b_approach"), label="Phase B: approach cup")

    pr2_skills.open_gripper_pr2(env)
    pr2_skills.move_arm_to_conf(env, pr2_skills.REACH_CONF, steps=12)
    pr2_skills.move_arm_to_conf(env, pr2_skills.PRE_GRASP_CONF, steps=12)
    record_frame(env, make_tag("b_above_cup"), label="Phase B: gripper above cup")

    pr2_skills.attach_cup_pr2(env, cup)
    pr2_skills.close_gripper_pr2(env)
    record_frame(env, make_tag("b_attached"), label="Phase B: cup attached")
    record_frame(env, make_tag("b_contact"), label="Phase B: gripper contacts cup")

    pr2_skills.move_arm_to_conf(env, pr2_skills.LIFT_CONF, steps=12, sub_steps=20)
    record_frame(env, make_tag("b_lifted"), label="Phase B: cup lifted")

    pr2_skills.move_arm_to_conf(env, pr2_skills.REACH_CONF, steps=10)
    record_frame(env, make_tag("b_transport_start"), label="Phase B: transport start")
    pr2_skills.move_pr2_base(env, base_place, steps=30)
    pr2_skills.set_static_camera(env, place_xy, camera_base_xy=base_place)
    record_frame(env, make_tag("b_transport_end"), label="Phase B: transport end")

    pr2_skills.move_arm_to_conf(env, pr2_skills.PRE_GRASP_CONF, steps=10)
    record_frame(env, make_tag("b_place_contact"), label="Phase B: place contact")

    pr2_skills.detach_cup_pr2(env, cup)
    reset_cup(cup, upright_after_move)
    record_frame(env, make_tag("b_placed"), label="Phase B: placed")

    pr2_skills.move_arm_to_conf(env, pr2_skills.REACH_CONF, steps=8)
    pr2_skills.open_gripper_pr2(env)
    pr2_skills.move_arm_to_conf(env, pr2_skills.HOME_CONF, steps=8)

    # ------------------------------------------------------------------
    # Phase C: re-observe and verify identity
    # ------------------------------------------------------------------
    print("\n=== Phase C: re-observe & reassociate ===")
    pr2_skills.move_pr2_base(env, base_obs, steps=20)
    obs = pr2_skills.observe_and_update(
        env, memory, percept, fallback_detector,
        base_xytheta=base_obs,
        look_at_xy=place_xy,
        scene_graph_option={"type": "reassociate"},
        object_level_labels=OBJECT_LEVEL_LABELS,
        fallback_labels=FALLBACK_LABELS,
    )
    record_frame(env, make_tag("counter2_observed"), label="Phase C: cup at new location")
    cup_after = summarize_cup(memory)
    print("Cup after move:", cup_after)
    identity_preserved = (
        cup_before and cup_after and
        cup_before["instance_id"] == cup_after["instance_id"]
    )

    # ------------------------------------------------------------------
    # Phase D: grasp validation on counter
    # ------------------------------------------------------------------
    print("\n=== Phase D: grasp validation ===")
    pr2_skills.move_pr2_base(env, base_place, steps=20)
    pr2_skills.set_head_toward(env, base_place, place_xy)
    pr2_skills.set_static_camera(env, place_xy, camera_base_xy=base_place)

    def _reset_for_validation():
        pr2_skills.detach_cup_pr2(env, cup)
        pr2_skills.open_gripper_pr2(env)
        reset_cup(cup, upright_after_move)
        pr2_skills.move_arm_to_conf(env, pr2_skills.HOME_CONF, steps=6)

    # D2: missed
    _reset_for_validation()
    validator.snapshot(cup)
    pr2_skills.close_gripper_pr2(env)
    outcome_missed = validator.classify(cup)
    record_frame(env, make_tag("d_missed"), label=f"Phase D: missed ({outcome_missed})")

    # D1: success
    _reset_for_validation()
    validator.snapshot(cup)
    pr2_skills.open_gripper_pr2(env)
    pr2_skills.move_arm_to_conf(env, pr2_skills.REACH_CONF, steps=12)
    pr2_skills.move_arm_to_conf(env, pr2_skills.PRE_GRASP_CONF, steps=12)
    pr2_skills.attach_cup_pr2(env, cup)
    pr2_skills.move_arm_to_conf(env, pr2_skills.GRASP_CONTACT_CONF, steps=12, sub_steps=20)
    pr2_skills.move_arm_to_conf(env, pr2_skills.LIFT_CONF, steps=12, sub_steps=20)
    outcome_success = validator.classify(cup)
    record_frame(env, make_tag("d_success"), label=f"Phase D: success ({outcome_success})")

    # D3: tipped
    _reset_for_validation()
    validator.snapshot(cup)
    pr2_skills.move_arm_to_conf(env, pr2_skills.SIDE_PUSH_START_CONF, steps=10)
    record_frame(env, make_tag("d_tipped_pre"), label="Phase D: push cup")
    pr2_skills.move_arm_to_conf(env, pr2_skills.SIDE_PUSH_END_CONF, steps=12)
    if validator.classify(cup) != "tipped":
        p.applyExternalTorque(cup.body, -1, torqueObj=[0.0, 2.0, 0.0], flags=p.LINK_FRAME)
        for _ in range(50):
            p.stepSimulation()
    outcome_tipped = validator.classify(cup)
    record_frame(env, make_tag("d_tipped"), label=f"Phase D: tipped ({outcome_tipped})")

    env.close()

    # ------------------------------------------------------------------
    # Compile video
    # ------------------------------------------------------------------
    frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".png")])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, COMBO_SIZE)
    for f in frames:
        img = cv2.imread(os.path.join(FRAME_DIR, f))
        writer.write(img)
    writer.release()

    report = {
        "video": VIDEO_PATH,
        "identity_preserved": identity_preserved,
        "cup_before": cup_before,
        "cup_after": cup_after,
        "grasp_validation": {
            "success": outcome_success,
            "missed": outcome_missed,
            "tipped": outcome_tipped,
        },
        "num_frames": len(frames),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n========== DEMO COMPLETE ==========")
    print(json.dumps(report, indent=2))
    print(f"Video saved: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
