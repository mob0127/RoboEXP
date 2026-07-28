# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Experiment: Validate grasp outcomes (success / missed / tipped) in PyBullet.

The FEG gripper is moved over a red cylinder (`cup`) on the counter.  With
``physical_grasp=True`` the gripper fingers are actuated and a fixed constraint
is created on ``close gripper`` when an object is near the tool center.  A
``GraspValidator`` records the pre-grasp state and classifies the post-grasp
state into one of {success, missed, tipped}.

NOTE: This is a PyBullet/simulation-only validation demo.  It relies on
PyBullet ground-truth poses and constraints, so it does not transfer to a real
robot.
"""
import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.validation.grasp_validator import GraspValidator

from pybullet_tools.utils import (
    PI, get_aabb, set_pose, set_mass, Pose, Point, create_cylinder, RED
)
from world_builder.entities import Object
import pybullet as p

CYLINDER_RADIUS = 0.08
CYLINDER_HEIGHT = 0.20

OUT_DIR = "/home/jx/kitchen/RoboEXP/experiments/outputs"
REPORT_PATH = os.path.join(OUT_DIR, "task5_cup_grasp/cup_pick_validate_report.json")
FRAME_DIR = os.path.join(OUT_DIR, "task5_cup_grasp/frames")
SAVE_FRAMES = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_frame(env, tag, global_view=False):
    """Save the current wrist-camera or global RGB as a demo frame."""
    if not SAVE_FRAMES:
        return
    ensure_dir(FRAME_DIR)
    if global_view:
        img = env.get_global_image()
    else:
        obs = env.get_observations(wrist_only=True)["wrist"]
        img = (obs["rgb"] * 255).clip(0, 255).astype(np.uint8)
    path = os.path.join(FRAME_DIR, f"{tag}.png")
    import cv2
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  [frame] {path}")


def add_test_cup(env, counter_top_z, xy, name="test_cup"):
    """Create a bright red cylinder on the counter and make it movable."""
    z = counter_top_z + CYLINDER_HEIGHT / 2 + 0.01
    body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
    cup = env.world.add_object(Object(body, category="cup", name=name))
    set_pose(cup.body, Pose(point=Point(xy[0], xy[1], z)))
    # Give the cup a non-zero mass so the fixed constraint can actually move it.
    set_mass(cup.body, 0.1)
    for _ in range(20):
        p.stepSimulation()
    # Use the actual AABB center for gripper targeting (the set_pose point may
    # differ from the geometry center depending on the collision shape origin).
    aabb = get_aabb(cup.body)
    center = (np.array(aabb[0]) + np.array(aabb[1])) / 2.0
    print(f"Added cup '{cup.name}' body={cup.body} at aabb center {center.tolist()}")
    return cup, center


def reset_cup(cup, pose):
    """Reset cup to the given upright pose and zero its velocity."""
    set_pose(cup.body, pose)
    p.resetBaseVelocity(cup.body, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    for _ in range(20):
        p.stepSimulation()


def move_gripper(env, xyz, roll=0.0, pitch=-PI / 2, yaw=0.0, iteration=50):
    """Move the FEG gripper to a target SE(3) pose."""
    env.run_action(1, [xyz[0], xyz[1], xyz[2], roll, pitch, yaw], iteration=iteration)


def test_success(env, validator, cup, cup_center):
    """Top-down grasp and lift.  Expected outcome: success."""
    print("\n=== TEST: success ===")
    validator.snapshot(cup)

    half_h = CYLINDER_HEIGHT / 2.0
    pre_grasp = cup_center + np.array([0, 0, half_h + 0.20])
    grasp = cup_center + np.array([0, 0, half_h + 0.06])
    lift = cup_center + np.array([0, 0, half_h + 0.25])

    save_frame(env, "task5_success_01_pre_grasp", global_view=True)
    move_gripper(env, pre_grasp)
    env.run_action(2, [], iteration=50)   # open
    move_gripper(env, grasp)
    save_frame(env, "task5_success_02_at_cup", global_view=True)
    env.run_action(3, [], iteration=50)   # close -> attach
    move_gripper(env, lift)
    save_frame(env, "task5_success_03_lifted", global_view=True)

    outcome = validator.classify(cup)
    report = validator.get_report(cup)
    print(json.dumps(report, indent=2))
    assert outcome == "success", f"Expected success, got {outcome}"
    return report


def test_missed(env, validator, cup, cup_center):
    """Close the gripper next to the cup.  Expected outcome: missed."""
    print("\n=== TEST: missed ===")
    validator.snapshot(cup)

    half_h = CYLINDER_HEIGHT / 2.0
    offset = cup_center + np.array([0.25, 0, half_h + 0.20])
    offset_down = cup_center + np.array([0.25, 0, half_h + 0.06])

    save_frame(env, "task5_missed_01_pre_grasp", global_view=True)
    move_gripper(env, offset)
    env.run_action(2, [], iteration=50)   # open
    move_gripper(env, offset_down)
    save_frame(env, "task5_missed_02_at_air", global_view=True)
    env.run_action(3, [], iteration=50)   # close -> nothing to attach
    move_gripper(env, offset)
    save_frame(env, "task5_missed_03_retracted", global_view=True)

    outcome = validator.classify(cup)
    report = validator.get_report(cup)
    print(json.dumps(report, indent=2))
    assert outcome == "missed", f"Expected missed, got {outcome}"
    return report


def test_tipped(env, validator, cup, cup_center):
    """Push the cup from the side near its top so that it topples."""
    print("\n=== TEST: tipped ===")
    validator.snapshot(cup)

    # Open gripper, move to the side near the top rim, then push through.
    env.run_action(2, [], iteration=50)
    side = cup_center + np.array([0, -0.18, CYLINDER_HEIGHT / 4])
    push_through = cup_center + np.array([0, 0.10, CYLINDER_HEIGHT / 4])

    save_frame(env, "task5_tipped_01_pre_push", global_view=True)
    move_gripper(env, side)
    save_frame(env, "task5_tipped_02_at_side", global_view=True)
    env.run_action(1, list(push_through) + [0, -PI / 2, 0], iteration=150)
    save_frame(env, "task5_tipped_03_after_push", global_view=True)

    outcome = validator.classify(cup)
    report = validator.get_report(cup)
    print(json.dumps(report, indent=2))
    assert outcome == "tipped", f"Expected tipped, got {outcome}"
    return report


def main():
    ensure_dir(os.path.dirname(REPORT_PATH))

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

    counter = env.world.name_to_object("counter")
    counter_top_z = get_aabb(counter.body)[1][2]
    cup_start_xy = [0.60, 0.00]
    cup, cup_center = add_test_cup(env, counter_top_z, cup_start_xy)
    upright_pose = Pose(point=Point(*cup_center))

    validator = GraspValidator(env, lift_threshold=0.03, tip_threshold=0.7)

    results = {}

    # 1. Success
    results["success"] = test_success(env, validator, cup, cup_center)

    # Reset cup for the next test.
    env.run_action(2, [], iteration=50)   # open -> detach
    reset_cup(cup, upright_pose)

    # 2. Missed
    results["missed"] = test_missed(env, validator, cup, cup_center)

    # Reset cup for the next test.
    env.run_action(2, [], iteration=50)
    reset_cup(cup, upright_pose)

    # 3. Tipped
    results["tipped"] = test_tipped(env, validator, cup, cup_center)

    env.close()

    summary = {
        "cup_start_position": cup_center.tolist(),
        "tests": results,
        "all_passed": all(r["outcome"] == name for name, r in results.items()),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved: {REPORT_PATH}")
    print(json.dumps(summary, indent=2))
    if summary["all_passed"]:
        print("\nAll grasp-validation tests passed.")
    else:
        print("\nSome grasp-validation tests FAILED.")


if __name__ == "__main__":
    main()
