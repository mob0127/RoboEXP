# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Dynamic end-to-end demo of tasks 3-5.

Shows a single continuous experiment:
  1. Observe a red cup on the shelf (with PyBullet fallback mask).
  2. The FEG gripper picks the cup, moves it to the counter, and places it.
  3. Re-observe and show that the cup keeps the same scene-graph identity.
  4. Demonstrate grasp validation: success / missed / tipped.

Every frame contains BOTH the global (bird's-eye) view and the wrist-camera
view, side-by-side, so the viewer can see the overall scene and the robot's
local perception at the same time.

Outputs:
  - outputs/dynamic_demo/dynamic_demo.mp4
  - outputs/dynamic_demo/frames/*.png
  - outputs/dynamic_demo/report.json
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
from roboexp.memory.robo_memory import RoboMemory
from roboexp.perception.robo_percept import RoboPercept
from roboexp.perception.pybullet_fallback import PyBulletFallbackDetector
from roboexp.validation.grasp_validator import GraspValidator

from pybullet_tools.utils import (
    PI, get_aabb, set_pose, set_mass, Pose, Point, create_cylinder, RED
)
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

OUT_DIR = "/home/jx/kitchen/RoboEXP/experiments/outputs/dynamic_demo"
FRAME_DIR = os.path.join(OUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUT_DIR, "dynamic_demo.mp4")
REPORT_PATH = os.path.join(OUT_DIR, "report.json")

FRAME_SIZE = (640, 480)          # each view
COMBO_SIZE = (1280, 480)         # side-by-side
FPS = 2
FRAME_COUNTER = 0


def make_tag(prefix):
    global FRAME_COUNTER
    FRAME_COUNTER += 1
    return f"f{FRAME_COUNTER:03d}_{prefix}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_test_cup(env, surface_top_z, xy, name="test_cup"):
    z = surface_top_z + CYLINDER_HEIGHT / 2 + 0.01
    body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
    cup = env.world.add_object(Object(body, category="cup", name=name))
    set_pose(cup.body, Pose(point=Point(xy[0], xy[1], z)))
    set_mass(cup.body, 0.1)
    for _ in range(20):
        p.stepSimulation()
    return cup


def reset_cup(cup, pose):
    set_pose(cup.body, pose)
    p.resetBaseVelocity(cup.body, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    for _ in range(20):
        p.stepSimulation()


def move_gripper(env, xyz, rpy=None, iteration=50):
    if rpy is None:
        rpy = [0, -PI / 2, 0]
    env.run_action(1, [xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]], iteration=iteration)


def get_gripper_pose(env):
    return np.array(env.get_end_effector_pose(), dtype=np.float32)


def move_gripper_linear(env, target_xyz, target_rpy=None, steps=6, iteration_per_step=20,
                        record_prefix=None, label=None, sublabel=None):
    """Move the gripper in small increments and capture frames along the way."""
    if target_rpy is None:
        target_rpy = np.array([0, -PI / 2, 0], dtype=np.float32)
    current = get_gripper_pose(env)
    target = np.concatenate([target_xyz, target_rpy])
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = current * (1 - alpha) + target * alpha
        env.run_action(1, list(interp), iteration=iteration_per_step)
        if record_prefix is not None:
            record_frame(env, make_tag(record_prefix), label=label, sublabel=sublabel)


def red_object_mask(rgb):
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 0.25) & (red > green + 0.15) & (red > blue + 0.15)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def observe_and_update(env, memory, percept, fallback_detector, scene_graph_option=None,
                       stage_name="OBSERVE"):
    """Move to a front viewpoint, capture wrist obs, update memory/scene graph."""
    viewpoint = [1.5, 0.0, 1.8, 0, -PI / 2.5, 0]
    move_gripper_linear(env, np.array(viewpoint[:3]), np.array(viewpoint[3:]),
                        steps=5, iteration_per_step=20,
                        record_prefix=f"move_to_view_{stage_name.lower()}",
                        label=f"move to {stage_name} view")
    observations = env.get_observations(wrist_only=True)
    obs_attrs = percept.get_attributes_with_fallback(
        observations, fallback_detector, FALLBACK_LABELS, replace_existing=True
    )
    memory.update_memory(
        observations, obs_attrs, OBJECT_LEVEL_LABELS,
        update_scene_graph=True, scene_graph_option=scene_graph_option,
        filter_masks={},
    )
    return observations["wrist"]


def record_frame(env, tag, label=None, sublabel=None, mask=None):
    """Capture global + wrist views, compose side-by-side, save frame and video."""
    ensure_dir(FRAME_DIR)
    wrist = env.get_observations(wrist_only=True)["wrist"]["rgb"]
    global_rgb = env.get_global_image(width=FRAME_SIZE[0], height=FRAME_SIZE[1])

    # Resize/normalize
    wrist_img = (wrist * 255).clip(0, 255).astype(np.uint8)
    wrist_img = cv2.resize(wrist_img, FRAME_SIZE)
    global_img = np.ascontiguousarray(global_rgb)
    global_img = cv2.resize(global_img, FRAME_SIZE)

    # Optional mask overlay on wrist view (resize mask to match wrist)
    if mask is not None:
        mask_r = cv2.resize(mask.astype(np.uint8), FRAME_SIZE).astype(bool)
        overlay = wrist_img.copy()
        overlay[mask_r] = (255, 0, 0)
        wrist_img = cv2.addWeighted(wrist_img, 0.5, overlay, 0.5, 0)

    # Side-by-side
    combo = np.hstack([global_img, wrist_img])

    # Labels on each view
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo, "GLOBAL", (20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, "WRIST", (FRAME_SIZE[0] + 20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

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
    # Remove stale frames from previous runs
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
        robot_name="feg",
        use_gui=False,
        segment=False,
        initial_base_q=(1.5, 0.0, 1.8),
        physical_grasp=True,
    )
    env.robot_move_to_pose([1.5, 0.0, 1.8, 0, -PI / 2.5, 0])
    env.run_action(0, [], iteration=50)
    record_frame(env, make_tag("setup"), label="Setup: kitchen scene")

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

    shelf = env.world.name_to_object("shelf")
    counter = env.world.name_to_object("counter")
    shelf_top = get_aabb(shelf.body)[1][2]
    counter_top = get_aabb(counter.body)[1][2]

    # ------------------------------------------------------------------
    # Phase A: cup on shelf
    # ------------------------------------------------------------------
    print("\n=== Phase A: cup on shelf ===")
    cup = add_test_cup(env, shelf_top, [0.25, 0.00], name="test_cup")
    upright_on_counter = Pose(point=Point(0.60, 0.00, counter_top + CYLINDER_HEIGHT / 2 + 0.01))

    obs = observe_and_update(env, memory, percept, fallback_detector,
                             scene_graph_option=None, stage_name="SHELF")
    mask = red_object_mask(obs["rgb"])
    record_frame(env, make_tag("shelf_initial"), label="Phase A: cup on shelf",
                 sublabel="fallback mask (red)", mask=mask)
    cup_before = summarize_cup(memory)
    print("Cup before move:", cup_before)

    # ------------------------------------------------------------------
    # Phase B: pick cup from shelf and place on counter
    # ------------------------------------------------------------------
    print("\n=== Phase B: pick & place ===")
    aabb = get_aabb(cup.body)
    cup_center = (np.array(aabb[0]) + np.array(aabb[1])) / 2.0
    half_h = CYLINDER_HEIGHT / 2.0

    # Approach from front of shelf
    pre_grasp = cup_center + np.array([0.0, 0.0, half_h + 0.25])
    grasp = cup_center + np.array([0.0, 0.0, half_h + 0.06])
    lift = cup_center + np.array([0.0, 0.0, half_h + 0.40])

    move_gripper_linear(env, pre_grasp, steps=4, record_prefix="b_approach",
                        label="Phase B: approach cup")
    env.run_action(2, [], iteration=50)   # open
    move_gripper_linear(env, grasp, steps=3, record_prefix="b_descend",
                        label="Phase B: descend to cup")
    env.run_action(3, [], iteration=50)   # close -> attach
    record_frame(env, make_tag("b_attached"), label="Phase B: gripper closed (attached)")
    move_gripper_linear(env, lift, steps=4, record_prefix="b_lift",
                        label="Phase B: lift cup")

    # Transport to counter
    place_xyz = np.array([0.60, 0.00, counter_top + half_h + 0.06])
    place_pre = place_xyz + np.array([0, 0, 0.25])
    move_gripper_linear(env, place_pre, steps=6, record_prefix="b_transport",
                        label="Phase B: transport to counter")
    move_gripper_linear(env, place_xyz, steps=4, record_prefix="b_place_descend",
                        label="Phase B: descend to counter")
    env.run_action(2, [], iteration=50)   # open -> detach
    record_frame(env, make_tag("b_placed"), label="Phase B: placed on counter")
    move_gripper_linear(env, place_pre, steps=3, record_prefix="b_retract",
                        label="Phase B: retract")

    # ------------------------------------------------------------------
    # Phase C: re-observe and verify identity
    # ------------------------------------------------------------------
    print("\n=== Phase C: re-observe & reassociate ===")
    obs = observe_and_update(env, memory, percept, fallback_detector,
                             scene_graph_option={"type": "reassociate"},
                             stage_name="COUNTER")
    record_frame(env, make_tag("counter_observed"), label="Phase C: re-observe cup on counter")
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
    # Reset cup upright on counter for clean validation sequence
    env.run_action(2, [], iteration=50)
    reset_cup(cup, upright_on_counter)
    aabb = get_aabb(cup.body)
    cup_center = (np.array(aabb[0]) + np.array(aabb[1])) / 2.0

    # D1: success
    validator.snapshot(cup)
    pre = cup_center + np.array([0, 0, half_h + 0.20])
    gr = cup_center + np.array([0, 0, half_h + 0.06])
    lift_pos = cup_center + np.array([0, 0, half_h + 0.25])
    move_gripper_linear(env, pre, steps=3, record_prefix="d_success_approach",
                        label="Phase D: success - approach")
    env.run_action(2, [], iteration=50)
    move_gripper_linear(env, gr, steps=2, record_prefix="d_success_descend",
                        label="Phase D: success - descend")
    env.run_action(3, [], iteration=50)
    move_gripper_linear(env, lift_pos, steps=3, record_prefix="d_success_lift",
                        label="Phase D: success - lift")
    outcome_success = validator.classify(cup)
    record_frame(env, make_tag("d_success_result"), label=f"Phase D: success ({outcome_success})")

    # Reset
    env.run_action(2, [], iteration=50)
    reset_cup(cup, upright_on_counter)

    # D2: missed
    validator.snapshot(cup)
    offset = cup_center + np.array([0.25, 0, half_h + 0.20])
    offset_down = cup_center + np.array([0.25, 0, half_h + 0.06])
    move_gripper_linear(env, offset, steps=3, record_prefix="d_missed_approach",
                        label="Phase D: missed - approach empty air")
    env.run_action(2, [], iteration=50)
    move_gripper_linear(env, offset_down, steps=2, record_prefix="d_missed_descend",
                        label="Phase D: missed - descend")
    env.run_action(3, [], iteration=50)
    move_gripper_linear(env, offset, steps=2, record_prefix="d_missed_retract",
                        label="Phase D: missed - retract")
    outcome_missed = validator.classify(cup)
    record_frame(env, make_tag("d_missed_result"), label=f"Phase D: missed ({outcome_missed})")

    # Reset
    env.run_action(2, [], iteration=50)
    reset_cup(cup, upright_on_counter)

    # D3: tipped
    validator.snapshot(cup)
    side = cup_center + np.array([0, -0.20, half_h * 0.5])
    push_through = cup_center + np.array([0, 0.25, half_h * 0.5])
    env.run_action(2, [], iteration=50)
    move_gripper_linear(env, side, steps=3, record_prefix="d_tipped_approach",
                        label="Phase D: tipped - approach side")
    record_frame(env, make_tag("d_tipped_push"), label="Phase D: tipped - push cup")
    env.run_action(1, list(push_through) + [0, -PI / 2, 0], iteration=150)
    # If the cup is still upright, give it a small torque impulse to topple.
    if validator.classify(cup) != "tipped":
        p.applyExternalTorque(cup.body, -1, torqueObj=[0.0, 2.0, 0.0],
                              flags=p.LINK_FRAME)
        for _ in range(50):
            p.stepSimulation()
    outcome_tipped = validator.classify(cup)
    record_frame(env, make_tag("d_tipped_result"), label=f"Phase D: tipped ({outcome_tipped})")

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
