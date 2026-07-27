"""
Reusable PR2 motion primitives for TAMP execution in PyBullet/kitchen-worlds.

These functions were extracted from ``experiments/cup_dynamic_demo_pr2.py`` so
that both demo scripts and TAMP executors can call the same PR2 skills.

All functions assume ``env`` is a ``PyBulletExplorationEnv`` with
``robot_name="pr2"``.
"""

import numpy as np
import pybullet as p

from pybullet_tools.utils import (
    PI, get_pose, get_link_pose, point_from_pose, Pose, Point,
    multiply, link_from_name, get_camera_matrix
)
from pybullet_tools.pr2_utils import (
    set_group_conf, get_arm_joints, set_arm_conf, open_arm, close_arm,
    PR2_TOOL_FRAMES, SIDE_HOLDING_LEFT_ARM
)
from world_builder.entities import StaticCamera


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
                      0.3691640376021203]

DEFAULT_ARM_CONFS = {
    "home": HOME_CONF,
    "reach": REACH_CONF,
    "pre_grasp": PRE_GRASP_CONF,
    "grasp_contact": GRASP_CONTACT_CONF,
    "lift": LIFT_CONF,
    "side_push_start": SIDE_PUSH_START_CONF,
    "side_push_end": SIDE_PUSH_END_CONF,
}


# ---------------------------------------------------------------------------
# Base movement
# ---------------------------------------------------------------------------

def get_base_xytheta(env):
    """Return the current PR2 base pose as [x, y, theta]."""
    pos, orn = p.getBasePositionAndOrientation(env.robot.body)
    yaw = p.getEulerFromQuaternion(orn)[2]
    return np.array([pos[0], pos[1], yaw])


def set_pr2_root(env, x, y, theta):
    """Snap the PR2 root to a planar pose and step the simulation briefly."""
    orn = p.getQuaternionFromEuler([0, 0, theta])
    p.resetBasePositionAndOrientation(env.robot.body, [x, y, 0.0], orn)
    for _ in range(10):
        p.stepSimulation()


def move_pr2_base(env, target_xytheta, steps=25):
    """Move the PR2 root smoothly by interpolating from the current pose.

    Uses small root resets rather than a motion planner.  If an object is
    attached, its pose is synced after each step.
    """
    cur = get_base_xytheta(env)
    target = np.asarray(target_xytheta, dtype=float)
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = cur * (1 - alpha) + target * alpha
        set_pr2_root(env, interp[0], interp[1], interp[2])
        if getattr(env, "grasped_body", None) is not None:
            sync_attached_cup(env)


# ---------------------------------------------------------------------------
# Head and camera
# ---------------------------------------------------------------------------

def set_head_toward(env, base_xytheta, look_at_xy):
    """Set head pan/tilt so the head camera looks at ``look_at_xy``."""
    dx = look_at_xy[0] - base_xytheta[0]
    dy = look_at_xy[1] - base_xytheta[1]
    pan = float(np.arctan2(dy, dx) - base_xytheta[2])
    pan = float(np.clip(pan, -2.8, 2.8))
    # Use a near-horizontal tilt so the head camera frames the cup above the arm.
    set_group_conf(env.robot.body, "head", [pan, 0.0])
    for _ in range(10):
        p.stepSimulation()


def set_pr2_viewpoint(env, base_xytheta, look_at_xy=None):
    """Set base pose and aim head at the target point."""
    set_pr2_root(env, base_xytheta[0], base_xytheta[1], base_xytheta[2])
    set_group_conf(env.robot.body, "torso", [0.25])
    if look_at_xy is not None:
        set_head_toward(env, base_xytheta, look_at_xy)
    else:
        set_group_conf(env.robot.body, "head", [0.0, 0.3])
        for _ in range(10):
            p.stepSimulation()


def set_static_camera(env, target_xy, camera_base_xy=None, camera_z=1.6):
    """Place an external StaticCamera looking at ``target_xy``.

    If ``camera_base_xy`` is given, the camera is placed just in front of the
    robot base (a robot-front/shoulder viewpoint).  Otherwise it falls back to
    a fixed external viewpoint.
    """
    target = [float(target_xy[0]), float(target_xy[1]), 1.43]
    if camera_base_xy is not None:
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


# ---------------------------------------------------------------------------
# Arm and gripper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Grasp attachment
# ---------------------------------------------------------------------------

def attach_cup_pr2(env, cup):
    """Create a fixed constraint between the left tool frame and ``cup``."""
    tool_link = link_from_name(env.robot.body, PR2_TOOL_FRAMES[ARM])
    env.grasped_cup = cup
    env.grasped_body = cup.body
    # Keep the cup centered on the tool frame so the closed gripper fingers
    # visibly surround the cup instead of the cup floating above the palm.
    env.grasp_rel_pose = Pose(point=Point(0, 0, 0))

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
    """Remove the fixed constraint and clear grasp state."""
    if getattr(env, "grasp_constraint", None) is not None:
        p.removeConstraint(env.grasp_constraint)
        env.grasp_constraint = None
    env.grasped_body = None
    env.grasped_cup = None
    env.grasp_rel_pose = None
    for _ in range(10):
        p.stepSimulation()


def sync_attached_cup(env):
    """Sync the attached cup pose to the left tool frame each step."""
    if getattr(env, "grasped_body", None) is None:
        return
    tool_link = link_from_name(env.robot.body, PR2_TOOL_FRAMES[ARM])
    tool_pose = get_link_pose(env.robot.body, tool_link)
    rel_point = point_from_pose(env.grasp_rel_pose)
    cup_pos = point_from_pose(multiply(tool_pose, Pose(point=Point(*rel_point))))
    p.resetBasePositionAndOrientation(env.grasped_body, [cup_pos[0], cup_pos[1], cup_pos[2]],
                                       p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBaseVelocity(env.grasped_body, [0, 0, 0], [0, 0, 0])


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def observe_and_update(
    env,
    memory,
    percept,
    fallback_detector,
    base_xytheta,
    look_at_xy=None,
    scene_graph_option=None,
    stage_name="OBSERVE",
    object_level_labels=None,
    fallback_labels=None,
    filter_masks=None,
):
    """Move PR2 to a viewpoint, capture observations, and update memory."""
    set_pr2_viewpoint(env, base_xytheta, look_at_xy=look_at_xy)
    observations = env.get_observations()

    if fallback_labels:
        observation_attributes = percept.get_attributes_with_fallback(
            observations, fallback_detector, fallback_labels, replace_existing=True
        )
    else:
        observation_attributes = percept.get_attributes_from_observations(
            observations, visualize=False
        )

    memory.update_memory(
        observations,
        observation_attributes,
        object_level_labels if object_level_labels is not None else [],
        filter_masks=filter_masks if filter_masks is not None else {},
        update_scene_graph=True,
        scene_graph_option=scene_graph_option,
        visualize=False,
    )
    return observations["head"]
