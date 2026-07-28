# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Closed-loop TAMP pick-and-place for PR2.

This script demonstrates the replanning controller: it observes the world,
plans with PDDLStream, executes one action at a time, observes again, and
replans if the world diverges from the planner's model.  A deliberate
perturbation is injected after the first ``move_base`` to exercise the
replanning path.

For now the state observer reads directly from PyBullet (``GTStateObserver``)
because the visual perception pipeline needs CUDA support that is not always
available in this environment.  The observer is abstract, so swapping in
``PerceptionObserver`` is a one-line change once the GPU runtime is healthy.
"""
import sys
import os
import json

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pddlstream")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

import random

import numpy as np
import cv2
import pybullet as p

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.env import pr2_skills
from roboexp.tamp.problem_builder import TAMPProblemBuilder
from roboexp.tamp.replanning_controller import (
    ReplanningTAMPController,
    GTStateObserver,
    # PerceptionObserver,  # swap this in once visual perception is available
)

from pybullet_tools.utils import (
    get_aabb, set_pose, set_mass, Pose, Point, create_cylinder, RED,
    create_box, BLUE, set_random_seed, get_pose, set_base_values,
)
from pybullet_tools.pr2_utils import set_group_conf, set_arm_conf
from world_builder.entities import Object


CYLINDER_RADIUS = 0.06
CYLINDER_HEIGHT = 0.12

TABLE_SIZE = (0.50, 0.50, 0.75)  # half-extents in x/y/z
TABLE_XY = (1.50, 1.50)
TABLE_COLOR = BLUE

OUT_DIR = "/home/jx/kitchen/RoboEXP/experiments/outputs/tamp_pick_place_pr2"
FRAME_DIR = os.path.join(OUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUT_DIR, "tamp_pick_place_pr2.mp4")
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


def add_test_cup(env, surface_top_z, xy, name="test_cup"):
    z = surface_top_z + CYLINDER_HEIGHT / 2 + 0.01
    body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
    cup = env.world.add_object(Object(body, category="cup", name=name))
    set_pose(cup.body, Pose(point=Point(xy[0], xy[1], z)))
    set_mass(cup.body, 0.1)
    for _ in range(50):
        p.stepSimulation()
    reset_cup(cup, Pose(point=Point(xy[0], xy[1], z)))
    return cup


def reset_cup(cup, pose):
    set_pose(cup.body, pose)
    p.resetBaseVelocity(cup.body, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    for _ in range(20):
        p.stepSimulation()


def record_frame(env, tag, label=None):
    """Save a global + robot-camera composite frame."""
    ensure_dir(FRAME_DIR)
    obs = env.get_observations()["head"]
    head_img = (obs["rgb"] * 255).clip(0, 255).astype(np.uint8)
    head_img = cv2.resize(head_img, FRAME_SIZE)
    global_img = np.ascontiguousarray(env.get_global_image(width=FRAME_SIZE[0], height=FRAME_SIZE[1]))
    combo = np.hstack([global_img, head_img])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo, "GLOBAL", (20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, "ROBOT", (FRAME_SIZE[0] + 20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(combo, label, (20, FRAME_SIZE[1] - 20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    path = os.path.join(FRAME_DIR, f"{tag}.png")
    cv2.imwrite(path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    return combo


def follow_base_path(env, path):
    """Follow a collision-free base path returned by plan-base-motion."""
    for bq in path:
        set_base_values(env.robot.body, bq)
        for _ in range(5):
            p.stepSimulation()
        if getattr(env, "grasped_body", None) is not None:
            pr2_skills.sync_attached_cup(env)


def move_to_ik_conf(env, bq, aq, protected_body=None):
    """Move the base and arm to the configuration certified by an IK stream.

    The arm transitions through a high retracted pose so it does not sweep
    across the counter and knock over objects.
    """
    pr2_skills.set_pr2_root(env, bq[0], bq[1], bq[2])
    pr2_skills.move_arm_via_conf_safe(
        env, aq, pr2_skills.REACH_CONF, protected_body=protected_body,
        margin=0.05, steps=12, sub_steps=10,
    )


def execute_action(env, action, perturb_after_first_move=True):
    """Execute a single PDDLStream action and optionally perturb the cup.

    The perturbation is a deliberate closed-loop test: after the first
    ``move_base`` we slide the cup 15 cm along the counter.  The next
    observation will show the cup out of place, the controller will detect the
    unexpected change, and it will replan from the new state.
    """
    if hasattr(action, "name"):
        name = action.name
        args = action.args
    else:
        name = action[0]
        args = action[1:]

    print(f"Executing: {name}")

    cup = _find_cup_from_world(env)
    protected_body = cup.body if cup is not None else None

    if name == "move_base":
        _robot, _q1, _q2, path = args
        follow_base_path(env, path)
        if perturb_after_first_move and not execute_action._perturbed:
            cup = _find_cup_from_world(env)
            if cup is not None:
                perturb_cup(env, cup, dx=0.15)
            execute_action._perturbed = True
        record_frame(env, make_tag("move"), label="move_base")

    elif name == "move_arm":
        _robot, _q1, q2 = args
        pr2_skills.move_arm_via_conf_safe(
            env, q2, pr2_skills.REACH_CONF, protected_body=protected_body,
            margin=0.05, steps=12, sub_steps=10,
        )
        record_frame(env, make_tag("arm"), label="move_arm")

    elif name == "pick":
        _robot, obj_body, _grasp, bq, aq, _pose = args
        cup = env.world.BODY_TO_OBJECT[obj_body]
        move_to_ik_conf(env, bq, aq, protected_body=cup.body)
        pr2_skills.close_gripper_pr2(env)
        pr2_skills.attach_cup_pr2(env, cup)
        record_frame(env, make_tag("pick"), label="pick cup")

    elif name == "place":
        _robot, obj_body, _grasp, bq, aq, pose, _surface = args
        cup = env.world.BODY_TO_OBJECT[obj_body]
        move_to_ik_conf(env, bq, aq, protected_body=cup.body)
        pr2_skills.detach_cup_pr2(env, cup)
        reset_cup(cup, pose)
        pr2_skills.open_gripper_pr2(env)
        record_frame(env, make_tag("place"), label="place cup")

    else:
        raise ValueError(f"Unknown action: {name}")


execute_action._perturbed = False


def _find_cup_from_world(env):
    for body, obj in env.world.BODY_TO_OBJECT.items():
        if obj.category.lower() == "cup":
            return obj
    return None


def perturb_cup(env, cup, dx=0.15):
    """Slide the cup on the counter by ``dx`` to simulate an external disturbance."""
    point, quat = get_pose(cup.body)
    new_x = float(point[0]) + dx
    new_y = float(point[1])
    new_z = float(point[2])
    new_pose = Pose(point=Point(new_x, new_y, new_z))
    print(f"[PERTURB] moving cup from ({point[0]:.2f},{point[1]:.2f}) to ({new_x:.2f},{new_y:.2f})")
    reset_cup(cup, new_pose)


def build_video_and_report(plan, cost, result):
    """Compile the recorded frames into a video and write a JSON report."""
    frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".png")])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, COMBO_SIZE)
    for f in frames:
        img = cv2.imread(os.path.join(FRAME_DIR, f))
        writer.write(img)
    writer.release()

    report = {
        "video": VIDEO_PATH,
        "target_surface": "target_table",
        "plan": [list(a) for a in plan],
        "cost": cost,
        "num_frames": len(frames),
        "replanning_result": result,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n========== TAMP PICK-PLACE COMPLETE ==========")
    print(json.dumps(report, indent=2))
    print(f"Video saved: {VIDEO_PATH}")


def main():
    random.seed(0)
    np.random.seed(0)
    set_random_seed(0)

    ensure_dir(OUT_DIR)
    ensure_dir(FRAME_DIR)
    for f in os.listdir(FRAME_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(FRAME_DIR, f))
    global FRAME_COUNTER
    FRAME_COUNTER = 0
    execute_action._perturbed = False

    print("==================== SETUP ====================")
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=False,
        initial_base_q=(0.0, 0.0, 0.0),
    )
    env.physical_grasp = True

    base_pick = np.array([1.50, 0.00, np.pi])
    pr2_skills.set_pr2_root(env, *base_pick)
    set_group_conf(env.robot.body, "torso", [0.25])
    set_group_conf(env.robot.body, "head", [0.0, 0.3])
    set_arm_conf(env.robot.body, "left", pr2_skills.HOME_CONF)
    env.run_action(0, [], iteration=50)

    counter = env.world.name_to_object("counter")
    counter_top = get_aabb(counter.body)[1][2]

    cup_xy = (0.65, 0.25)
    cup = add_test_cup(env, counter_top, cup_xy, name="test_cup")

    table_body = create_box(*TABLE_SIZE, color=TABLE_COLOR)
    table_z = TABLE_SIZE[2]
    set_pose(table_body, Pose(point=Point(TABLE_XY[0], TABLE_XY[1], table_z)))
    target_surface = env.world.add_object(
        Object(table_body, category="table", name="target_table")
    )

    # Return the robot to the canonical start pose before the run.
    pr2_skills.set_pr2_root(env, *base_pick)
    set_group_conf(env.robot.body, "torso", [0.25])
    set_group_conf(env.robot.body, "head", [0.0, 0.3])
    set_arm_conf(env.robot.body, "left", pr2_skills.HOME_CONF)
    env.run_action(0, [], iteration=50)

    record_frame(env, make_tag("setup"), label="TAMP setup -> target_table")

    observer = GTStateObserver(
        movable_categories={"cup"},
        surface_categories={"counter", "table"},
    )
    builder = TAMPProblemBuilder(
        env,
        arm="left",
        movable_categories={"cup"},
        surface_categories={"counter", "table"},
    )
    controller = ReplanningTAMPController(
        env,
        observer=observer,
        problem_builder=builder,
        max_replans=3,
    )

    goal = ("On", cup.body, target_surface.body)

    def on_replan(replan_idx, action, changes):
        tag = make_tag("replan")
        label = f"replan {replan_idx}: {action.name if action else 'goal-not-reached'}"
        record_frame(env, tag, label=label)
        print(f"[REPLAN {replan_idx}] triggered by {action}: {changes}")

    result = controller.run(
        goal=goal,
        execute_action_fn=lambda env, action: execute_action(env, action, perturb_after_first_move=False),
        on_replan_fn=on_replan,
    )

    # Append hold frames so the video loop does not snap from the final place
    # frame back to the setup frame.
    HOLD_FRAMES = 15
    for i in range(HOLD_FRAMES):
        record_frame(env, make_tag("hold"), label="place (hold)")

    env.close()

    # Estimate the final plan/cost from the result for the report.
    final_plan = result.get("final_plan", [])
    cost = float(len(final_plan)) if final_plan else 0.0
    build_video_and_report(final_plan, cost, result)


if __name__ == "__main__":
    main()
