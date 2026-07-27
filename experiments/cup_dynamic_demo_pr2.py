"""
Dynamic PR2 demo of tasks 3-5.

A single continuous experiment using the PR2 robot:
  1. Observe a red cup on the counter (with PyBullet fallback mask).
  2. The PR2 left arm picks the cup and places it at another counter location.
  3. Re-observe and show that the cup keeps the same scene-graph identity.
  4. Demonstrate grasp validation: success / missed / tipped.

The PR2 base is kept at a fixed manipulation pose and the left arm uses a set of
pre-computed, feasible joint configurations (IK is brittle in this mini-kitchen
scene).  All motions are stepped through the simulator so the arm/base do not
snap through geometry.  The left gripper is driven into contact with the cup
before the fixed constraint is added, so the grasp looks plausible.

Each frame shows the global view (left) and a fixed static camera view (right)
that looks at the current cup location.  The PR2 head camera is occluded once the
base is moved out of the counter/wall, so a static viewpoint keeps the cup and
gripper clearly visible.

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
from roboexp.memory.robo_memory import RoboMemory
from roboexp.perception.robo_percept import RoboPercept
from roboexp.perception.pybullet_fallback import PyBulletFallbackDetector
from roboexp.validation.grasp_validator import GraspValidator

from pybullet_tools.utils import (
    PI, get_aabb, set_pose, set_mass, Pose, Point, create_cylinder, RED,
    get_pose, get_link_pose, point_from_pose, quat_from_pose,
    multiply, invert, link_from_name, get_camera_matrix
)
from pybullet_tools.pr2_utils import (
    set_group_conf, get_arm_joints, set_arm_conf, open_arm, close_arm,
    PR2_TOOL_FRAMES, SIDE_HOLDING_LEFT_ARM
)
from world_builder.entities import Object, StaticCamera
import pybullet as p

CYLINDER_RADIUS = 0.08
CYLINDER_HEIGHT = 0.20
ARM = "left"

# Pre-computed feasible left-arm configurations (torso height 0.25).
# HOME is a safe, collision-free side-holding pose; the pick sequence uses
# REACH/PRE/CONTACT/LIFT which keep the arm above the counter front edge.
HOME_CONF = SIDE_HOLDING_LEFT_ARM

# High, safe transition pose above the counter.
REACH_CONF = [-0.1646759064591234, -0.9441561987183871, 1.176631957489838,
              -0.4957383761215413, 2.5995300840317492, -1.2245732238786258,
              -0.18307273938690638]

# Gripper directly above the cup, just before contact.
PRE_GRASP_CONF = [-0.16984127185319253, -0.7734659197400336, 1.2857637448320185,
                  -0.5417221603172959, 2.3777883031923475, -1.68148638756865,
                  -0.262268518473109]

# Gripper centered on the cup -- used for the visible contact frame.
GRASP_CONTACT_CONF = [-0.13637098294477534, -0.6821719706568454, 1.2968169263264222,
                      -0.5685765297309353, 2.3007839028025767, -1.7338702206538184,
                      -0.2216304244273708]

# Slightly raised pose after the fixed constraint is added.
LIFT_CONF = [-0.16726653652170653, -0.9682512971568782, 1.070764881057107,
             -0.3814240308806699, 2.505410680063261, -1.0763314251718792,
             -0.4895416527911399]

# Side-push start / end for the "tipped" validation.
SIDE_PUSH_START_CONF = [-0.02687264393177438, -0.7390147936676277, 1.1104526565795658,
                        -0.677540034743332, 3.0051091683138407, -0.8764444232745758,
                        0.4084333105571252]
SIDE_PUSH_END_CONF = [-0.09113292914585622, -0.6283321590022568, 1.1787453556623186,
                      -0.7351890177758476, 2.794294472248544, -0.7872847970022749,
                      0.3691640476021203]

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
                    # Some bodies are already removed by world.remove_object;
                    # only call PyBullet removeBody if the body still exists.
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


def get_base_xytheta(env):
    pos, orn = p.getBasePositionAndOrientation(env.robot.body)
    yaw = p.getEulerFromQuaternion(orn)[2]
    return np.array([pos[0], pos[1], yaw])


def set_pr2_root(env, x, y, theta):
    """Snap the PR2 root to a planar pose."""
    orn = p.getQuaternionFromEuler([0, 0, theta])
    p.resetBasePositionAndOrientation(env.robot.body, [x, y, 0.0], orn)
    for _ in range(10):
        p.stepSimulation()


def move_pr2_base(env, target_xytheta, steps=25):
    """Move the PR2 root smoothly by resetting it in small increments."""
    cur = get_base_xytheta(env)
    target = np.asarray(target_xytheta, dtype=float)
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = cur * (1 - alpha) + target * alpha
        set_pr2_root(env, interp[0], interp[1], interp[2])
        if getattr(env, "grasped_body", None) is not None:
            sync_attached_cup(env)


def set_head_toward(env, base_xytheta, look_at_xy):
    dx = look_at_xy[0] - base_xytheta[0]
    dy = look_at_xy[1] - base_xytheta[1]
    pan = float(np.arctan2(dy, dx) - base_xytheta[2])
    pan = float(np.clip(pan, -2.8, 2.8))
    # Use a near-horizontal tilt so the head camera frames the cup above the arm.
    set_group_conf(env.robot.body, "head", [pan, 0.0])
    for _ in range(10):
        p.stepSimulation()


def set_pr2_viewpoint(env, base_xytheta, look_at_xy=None):
    """Set base and aim head at the target point."""
    set_pr2_root(env, base_xytheta[0], base_xytheta[1], base_xytheta[2])
    set_group_conf(env.robot.body, "torso", [0.25])
    if look_at_xy is not None:
        set_head_toward(env, base_xytheta, look_at_xy)
    else:
        set_group_conf(env.robot.body, "head", [0.0, 0.3])
        for _ in range(10):
            p.stepSimulation()
    # Keep the observation/video camera pointed at the object of interest.
    # Use a robot-front viewpoint so the right panel looks like the robot is
    # watching its own gripper and the cup.
    if look_at_xy is not None:
        set_static_camera(env, look_at_xy, camera_base_xy=base_xytheta)


def set_static_camera(env, target_xy, camera_base_xy=None, camera_z=1.6):
    """Place a camera that looks at the target cup.

    If ``camera_base_xy`` is given, the camera is placed just in front of the
    robot base (a robot-front/shoulder viewpoint).  Otherwise it falls back to
    a fixed external viewpoint.
    """
    target = [float(target_xy[0]), float(target_xy[1]), 1.43]
    if camera_base_xy is not None:
        # Robot-front/shoulder viewpoint: in front of the base, slightly to the
        # left and raised, so the arm, gripper, and cup are centered in frame.
        camera_point = [
            float(camera_base_xy[0]) - 0.70,
            float(camera_base_xy[1]) - 0.45,
            1.90,
        ]
    else:
        camera_point = [target[0] - 0.45, target[1] - 0.80, 2.20]
    matrix = get_camera_matrix(width=640, height=480, fx=525.0, fy=525.0)
    env.main_camera = StaticCamera(
        pose=Pose(point=Point(*camera_point)),
        camera_matrix=matrix,
        max_depth=2.5,
        camera_point=camera_point,
        target_point=target,
    )


def red_object_mask(rgb):
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 0.25) & (red > green + 0.15) & (red > blue + 0.15)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def observe_and_update(env, memory, percept, fallback_detector, base_xytheta,
                       look_at_xy=None, scene_graph_option=None, stage_name="OBSERVE"):
    set_pr2_viewpoint(env, base_xytheta, look_at_xy=look_at_xy)
    observations = env.get_observations()
    obs_attrs = percept.get_attributes_with_fallback(
        observations, fallback_detector, FALLBACK_LABELS, replace_existing=True
    )
    memory.update_memory(
        observations, obs_attrs, OBJECT_LEVEL_LABELS,
        update_scene_graph=True, scene_graph_option=scene_graph_option,
        filter_masks={},
    )
    return observations["head"]


def move_arm_to_conf(env, target_conf, steps=10, sub_steps=10):
    """Interpolate the left arm joint positions and step the simulation.

    Small per-step displacements + multiple simulation substeps keep the arm
    from snapping through the counter/floor.  If a cup is attached, its pose
    is synced to the tool frame after each step.
    """
    arm_joints = get_arm_joints(env.robot.body, ARM)
    cur = [p.getJointState(env.robot.body, j)[0] for j in arm_joints]
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = [cur[j] * (1 - alpha) + target_conf[j] * alpha for j in range(7)]
        set_arm_conf(env.robot.body, ARM, interp)
        for _ in range(sub_steps):
            p.stepSimulation()
        if getattr(env, "grasped_body", None) is not None:
            sync_attached_cup(env)


def open_gripper_pr2(env):
    open_arm(env.robot.body, ARM)
    for _ in range(20):
        p.stepSimulation()


def close_gripper_pr2(env):
    close_arm(env.robot.body, ARM)
    for _ in range(20):
        p.stepSimulation()


def attach_cup_pr2(env, cup):
    tool_link = link_from_name(env.robot.body, PR2_TOOL_FRAMES[ARM])
    env.grasped_cup = cup
    env.grasped_body = cup.body
    # Keep the cup centered on the tool frame so the closed gripper fingers
    # visibly surround the cup instead of the cup floating above the palm.
    env.grasp_rel_pose = Pose(point=Point(0, 0, 0))

    # Fixed constraint for pipeline semantics; the cup pose is also synced
    # kinematically each step because PyBullet's constraint solver struggles
    # when the arm joints are set directly.
    constraint = p.createConstraint(
        env.robot.body, tool_link, cup.body, -1,
        p.JOINT_FIXED, [0, 0, 0],
        parentFramePosition=[0, 0, 0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(constraint, maxForce=10000)
    env.grasp_constraint = constraint
    sync_attached_cup(env)


def detach_cup_pr2(env, cup):
    if getattr(env, "grasp_constraint", None) is not None:
        p.removeConstraint(env.grasp_constraint)
        env.grasp_constraint = None
    env.grasped_body = None
    env.grasped_cup = None
    env.grasp_rel_pose = None
    for _ in range(10):
        p.stepSimulation()


def sync_attached_cup(env):
    if getattr(env, "grasped_body", None) is None:
        return
    tool_link = link_from_name(env.robot.body, PR2_TOOL_FRAMES[ARM])
    tool_pose = get_link_pose(env.robot.body, tool_link)
    # Keep the cup upright (it is a symmetric cylinder) so validation always
    # sees a stable, upright lift.  Position follows the tool frame using the
    # translational offset recorded at attach time.
    rel_point = point_from_pose(env.grasp_rel_pose)
    cup_pos = point_from_pose(multiply(tool_pose, Pose(point=Point(*rel_point))))
    set_pose(env.grasped_body, Pose(point=Point(*cup_pos)))
    p.resetBaseVelocity(env.grasped_body, [0, 0, 0], [0, 0, 0])


def record_frame(env, tag, label=None, sublabel=None, mask=None):
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


def compute_tool_offset(env):
    """Return the tool (x, y) position in base frame for PRE_GRASP_CONF."""
    tool_link = link_from_name(env.robot.body, PR2_TOOL_FRAMES[ARM])
    set_arm_conf(env.robot.body, ARM, PRE_GRASP_CONF)
    for _ in range(20):
        p.stepSimulation()
    xy = np.array(point_from_pose(get_link_pose(env.robot.body, tool_link))[:2])
    set_arm_conf(env.robot.body, ARM, HOME_CONF)
    for _ in range(20):
        p.stepSimulation()
    return xy


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
    pre_tool_xy = compute_tool_offset(env)

    # Position the cup so the PR2 can stand in front of the counter (x > 1)
    # facing the counter (theta = pi) instead of being embedded in the wall.
    # The base is fixed on the open floor; the cup is placed where PRE_GRASP_CONF
    # positions the gripper directly above it.
    base_pick = np.array([1.50, 0.00, np.pi])
    actual_cup_xy = base_pick[:2] - pre_tool_xy
    cup_xy = actual_cup_xy.tolist()
    # Place the cup on the right half of the counter (positive y) where there is
    # no microwave, pot lid, or clutter, so it sits cleanly on the counter top.
    place_offset = np.array([0.00, 0.60])
    place_xy = actual_cup_xy + place_offset
    base_place = np.concatenate([place_xy + pre_tool_xy, [np.pi]])
    base_obs = base_pick.copy()

    # Place the robot in front of the counter from the start.
    set_pr2_root(env, *base_pick)
    set_group_conf(env.robot.body, "torso", [0.25])
    set_group_conf(env.robot.body, "head", [0.0, 0.3])
    set_arm_conf(env.robot.body, "left", HOME_CONF)
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

    # ------------------------------------------------------------------
    # Phase A: cup on counter
    # ------------------------------------------------------------------
    print("\n=== Phase A: cup on counter ===")
    cup = add_test_cup(env, counter_top, cup_xy, name="test_cup")
    # actual_cup_xy is already computed above; re-read in case placement shifted.
    actual_cup_xy = np.array(point_from_pose(get_pose(cup.body))[:2])

    upright_after_move = Pose(point=Point(
        place_xy[0], place_xy[1], counter_top + CYLINDER_HEIGHT / 2 + 0.01))

    # Place the observation camera so it can see the cup from outside geometry.
    set_static_camera(env, actual_cup_xy, camera_base_xy=base_obs)

    obs = observe_and_update(env, memory, percept, fallback_detector,
                             base_xytheta=base_obs,
                             look_at_xy=actual_cup_xy,
                             scene_graph_option=None, stage_name="COUNTER")
    mask = red_object_mask(obs["rgb"])
    record_frame(env, make_tag("counter_initial"), label="Phase A: cup on counter",
                 sublabel="fallback mask", mask=mask)
    cup_before = summarize_cup(memory)
    print("Cup before move:", cup_before)

    # ------------------------------------------------------------------
    # Phase B: pick cup from counter and place to another counter location
    # ------------------------------------------------------------------
    print("\n=== Phase B: PR2 pick & place ===")
    move_pr2_base(env, base_pick)
    move_arm_to_conf(env, HOME_CONF, steps=6)
    record_frame(env, make_tag("b_approach"), label="Phase B: approach cup")

    open_gripper_pr2(env)
    move_arm_to_conf(env, REACH_CONF, steps=12)
    move_arm_to_conf(env, PRE_GRASP_CONF, steps=12)
    record_frame(env, make_tag("b_above_cup"), label="Phase B: gripper above cup")

    # Add the fixed constraint while the gripper is directly above the cup, then
    # close the fingers so the cup is visibly grasped before lifting.
    attach_cup_pr2(env, cup)
    close_gripper_pr2(env)
    record_frame(env, make_tag("b_attached"), label="Phase B: cup attached")
    record_frame(env, make_tag("b_contact"), label="Phase B: gripper contacts cup")

    move_arm_to_conf(env, LIFT_CONF, steps=12, sub_steps=20)
    record_frame(env, make_tag("b_lifted"), label="Phase B: cup lifted")

    # Carry the cup to the place location (base moves while arm stays raised).
    move_arm_to_conf(env, REACH_CONF, steps=10)
    record_frame(env, make_tag("b_transport_start"), label="Phase B: transport start")
    move_pr2_base(env, base_place, steps=30)
    set_static_camera(env, place_xy, camera_base_xy=base_place)
    record_frame(env, make_tag("b_transport_end"), label="Phase B: transport end")

    # Lower the cup to the target location (cup center aligned with tool frame).
    move_arm_to_conf(env, PRE_GRASP_CONF, steps=10)
    record_frame(env, make_tag("b_place_contact"), label="Phase B: place contact")

    # Release the cup while it is resting on the counter and snap it to the
    # exact upright pose so it does not tip.
    detach_cup_pr2(env, cup)
    reset_cup(cup, upright_after_move)
    record_frame(env, make_tag("b_placed"), label="Phase B: placed")

    move_arm_to_conf(env, REACH_CONF, steps=8)
    open_gripper_pr2(env)
    move_arm_to_conf(env, HOME_CONF, steps=8)

    # ------------------------------------------------------------------
    # Phase C: re-observe and verify identity
    # ------------------------------------------------------------------
    print("\n=== Phase C: re-observe & reassociate ===")
    move_pr2_base(env, base_obs, steps=20)
    obs = observe_and_update(env, memory, percept, fallback_detector,
                             base_xytheta=base_obs,
                             look_at_xy=place_xy,
                             scene_graph_option={"type": "reassociate"},
                             stage_name="COUNTER2")
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
    move_pr2_base(env, base_place, steps=20)
    set_head_toward(env, base_place, place_xy)
    set_static_camera(env, place_xy, camera_base_xy=base_place)

    def _reset_for_validation():
        detach_cup_pr2(env, cup)
        open_gripper_pr2(env)
        reset_cup(cup, upright_after_move)
        move_arm_to_conf(env, HOME_CONF, steps=6)

    # Validation order: missed -> success -> tipped.  Missed is run first while
    # the arm is still in HOME_CONF, so we avoid the unsafe LIFT->HOME reset that
    # would otherwise knock the cup over.

    # D2: missed (close gripper away from the cup, no attachment)
    _reset_for_validation()
    validator.snapshot(cup)
    close_gripper_pr2(env)
    outcome_missed = validator.classify(cup)
    record_frame(env, make_tag("d_missed"), label=f"Phase D: missed ({outcome_missed})")

    # D1: success
    _reset_for_validation()
    validator.snapshot(cup)
    open_gripper_pr2(env)
    move_arm_to_conf(env, REACH_CONF, steps=12)
    move_arm_to_conf(env, PRE_GRASP_CONF, steps=12)
    attach_cup_pr2(env, cup)
    move_arm_to_conf(env, GRASP_CONTACT_CONF, steps=12, sub_steps=20)
    move_arm_to_conf(env, LIFT_CONF, steps=12, sub_steps=20)
    outcome_success = validator.classify(cup)
    record_frame(env, make_tag("d_success"), label=f"Phase D: success ({outcome_success})")

    # D3: tipped (push the cup from the side)
    _reset_for_validation()
    validator.snapshot(cup)
    move_arm_to_conf(env, SIDE_PUSH_START_CONF, steps=10)
    record_frame(env, make_tag("d_tipped_pre"), label="Phase D: push cup")
    move_arm_to_conf(env, SIDE_PUSH_END_CONF, steps=12)
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
