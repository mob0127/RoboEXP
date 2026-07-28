# Xiao : 新增文件，用于本仓库对上游的扩展。
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
    multiply, link_from_name, get_camera_matrix, get_closest_points
)
from pybullet_tools.pr2_utils import (
    set_group_conf, get_arm_joints, get_gripper_joints, set_arm_conf, open_arm, close_arm,
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

def move_arm_via_conf(env, target_conf, via_conf, steps=10, sub_steps=10):
    """Move the left arm through an intermediate ``via_conf`` to avoid sweeping
    through obstacles on the counter.

    The path is current -> via_conf -> target_conf.  Each segment uses the
    same small interpolation steps as ``move_arm_to_conf``.  Keeping the via
    pose high above the counter keeps the arm clear of objects while it
    transitions between workspace regions.
    """
    move_arm_to_conf(env, via_conf, steps=steps, sub_steps=sub_steps)
    move_arm_to_conf(env, target_conf, steps=steps, sub_steps=sub_steps)


def _arm_joint_values(env):
    arm_joints = get_arm_joints(env.robot.body, ARM)
    return [p.getJointState(env.robot.body, j)[0] for j in arm_joints]


def _set_arm_conf_safe(env, conf):
    """Set arm configuration for geometric checks (no physics stepping)."""
    set_arm_conf(env.robot.body, ARM, conf)


def _protected_body_for_clearance(env):
    """Return the body we must not hit while moving the free arm.

    For now this is the currently grasped cup if any, because the planner's
    certifies grasps and we want to avoid sweeping against it while the arm is
    still free.  Callers can override by passing ``protected_body`` explicitly.
    """
    return getattr(env, "grasped_body", None)


def _min_link_clearance(env, protected_body, max_distance=0.05, exclude_gripper=True):
    """Minimum distance between any robot arm/gripper link and ``protected_body``.

    Returns a float.  Positive means separation, negative means penetration.
    ``max_distance`` caps the search range; if no pair is closer than this,
    ``max_distance`` is returned.  Arm links plus gripper finger links are
    checked so that the open fingers cannot sweep the object while the arm is
    moving.
    """
    if protected_body is None:
        return float("inf")
    robot = env.robot.body
    arm_joints = get_arm_joints(env.robot.body, ARM)
    gripper_joints = get_gripper_joints(env.robot.body, ARM)
    # For non-base links in PyBullet URDFs, the link index equals the joint index.
    links = list(arm_joints) + list(gripper_joints)
    tool_link = link_from_name(robot, PR2_TOOL_FRAMES[ARM]) if exclude_gripper else None
    min_dist = max_distance
    for link in links:
        if exclude_gripper and link == tool_link:
            continue
        contacts = get_closest_points(robot, protected_body, link1=link, max_distance=max_distance)
        for c in contacts:
            if c.contactDistance < min_dist:
                min_dist = c.contactDistance
    return min_dist


def _is_joint_path_clear(env, q1, q2, protected_body, margin, steps=10):
    """Check whether a linear joint-space interpolation stays ``margin`` clear.

    The current arm configuration is saved and restored so the check is side-effect
    free.  All intermediate points must keep at least ``margin`` clearance; the
    final target point is allowed to be close (e.g. a pre-grasp pose) but must
    not penetrate.
    """
    if protected_body is None or protected_body == getattr(env, "grasped_body", None):
        return True
    arm_joints = get_arm_joints(env.robot.body, ARM)
    saved = _arm_joint_values(env)
    clear = True
    min_dist = float("inf")
    search_distance = max(0.05, margin + 0.02)
    for i in range(steps + 1):
        alpha = i / steps
        conf = [q1[j] * (1 - alpha) + q2[j] * alpha for j in range(7)]
        _set_arm_conf_safe(env, conf)
        d = _min_link_clearance(env, protected_body, max_distance=search_distance)
        min_dist = min(min_dist, d)
        if i < steps:
            if d < margin:
                clear = False
                break
        else:
            # Final pose may be intentionally close (pre-grasp).  Penetration
            # is still forbidden.
            if d < -0.005:
                clear = False
                break
    set_arm_conf(env.robot.body, ARM, saved)
    return clear


def _random_joint_perturbation(q, rng, scale=0.15):
    """Perturb a 7-DOF arm configuration, biasing the shoulder/elbow upward."""
    perturbed = list(q)
    # Shoulder lift and elbow flex are the main elevation DOFs.  More negative
    # shoulder lift generally raises the arm; more negative elbow flex folds it
    # up.  We sample a symmetric window and let the clearance test pick the best.
    perturbed[1] += rng.uniform(-scale, scale)      # shoulder lift
    perturbed[3] += rng.uniform(-scale, scale)      # elbow flex
    perturbed[0] += rng.uniform(-0.08, 0.08)        # shoulder pan
    perturbed[2] += rng.uniform(-0.10, 0.10)        # upper arm roll
    perturbed[4] += rng.uniform(-0.10, 0.10)        # forearm roll
    perturbed[5] += rng.uniform(-0.08, 0.08)        # wrist flex
    perturbed[6] += rng.uniform(-0.15, 0.15)        # wrist roll
    return perturbed


def _search_safe_via_configuration(env, q1, q2, protected_body, margin, attempts=30):
    """Search for an intermediate arm configuration that yields a clear two-segment path.

    We sample perturbations along the straight-line joint path between ``q1`` and
    ``q2`` and keep the candidate with the largest clearance for both
    ``q1->candidate`` and ``candidate->q2``.  If none beats ``margin``, return
    None.
    """
    if protected_body is None or protected_body == getattr(env, "grasped_body", None):
        return None
    rng = np.random.RandomState(0)
    best_via = None
    best_score = -float("inf")
    for _ in range(attempts):
        t = rng.uniform(0.25, 0.75)
        base_via = [q1[j] * (1 - t) + q2[j] * t for j in range(7)]
        via = _random_joint_perturbation(base_via, rng, scale=0.20)
        # Quick feasibility: check self clearance at the via pose first.
        _set_arm_conf_safe(env, via)
        via_clear = _min_link_clearance(env, protected_body, max_distance=0.08)
        if via_clear < margin:
            continue
        seg1_clear = _is_joint_path_clear(env, q1, via, protected_body, margin, steps=8)
        if not seg1_clear:
            continue
        seg2_clear = _is_joint_path_clear(env, via, q2, protected_body, margin, steps=8)
        if not seg2_clear:
            continue
        score = min(via_clear, margin + 0.01)
        if score > best_score:
            best_score = score
            best_via = via
    if best_via is not None:
        return best_via
    # Last resort: try a strongly elevated midpoint (shoulder up, elbow folded).
    elevated = [q1[j] * 0.5 + q2[j] * 0.5 for j in range(7)]
    elevated[1] -= 0.35
    elevated[3] -= 0.35
    elevated[5] -= 0.15
    _set_arm_conf_safe(env, elevated)
    if _min_link_clearance(env, protected_body, max_distance=0.08) >= margin:
        if _is_joint_path_clear(env, q1, elevated, protected_body, margin, steps=8) and \
           _is_joint_path_clear(env, elevated, q2, protected_body, margin, steps=8):
            return elevated
    return None


def move_arm_to_conf_safe(env, target_conf, protected_body=None, margin=0.05,
                          steps=12, sub_steps=10):
    """Move the left arm to ``target_conf`` while keeping a clearance margin.

    If ``protected_body`` is None, the currently grasped body is used when
    relevant (a grasped body is skipped because it is rigidly attached to the
    gripper).  The function first checks the straight-line joint path; if the
    predicted minimum clearance is below ``margin`` it searches for a safe via
    configuration in joint space.  If no safe via exists, the motion is executed
    very slowly with collision monitoring as a final fallback.
    """
    if protected_body is None:
        protected_body = _protected_body_for_clearance(env)
    # If the protected body is grasped, the fixed constraint handles it; just
    # execute normally, but still avoid other objects via the same path logic.
    if protected_body is not None and protected_body == getattr(env, "grasped_body", None):
        protected_body = None
    arm_joints = get_arm_joints(env.robot.body, ARM)
    current_conf = [p.getJointState(env.robot.body, j)[0] for j in arm_joints]
    if _is_joint_path_clear(env, current_conf, target_conf, protected_body, margin, steps=20):
        move_arm_to_conf(env, target_conf, steps=steps, sub_steps=sub_steps)
        return
    via = _search_safe_via_configuration(env, current_conf, target_conf, protected_body, margin)
    if via is not None:
        move_arm_to_conf(env, via, steps=steps, sub_steps=sub_steps)
        move_arm_to_conf(env, target_conf, steps=steps, sub_steps=sub_steps)
        return
    # Fallback: slow guarded execution.  We stop early if the body is hit.
    print(f"[WARN] No safe via found; executing guarded slow motion (margin={margin})")
    cur = current_conf
    for i in range(1, steps * 3 + 1):
        alpha = i / (steps * 3)
        interp = [cur[j] * (1 - alpha) + target_conf[j] * alpha for j in range(7)]
        set_arm_conf(env.robot.body, ARM, interp)
        for _ in range(sub_steps):
            p.stepSimulation()
        if getattr(env, "grasped_body", None) is not None:
            sync_attached_cup(env)
        d = _min_link_clearance(env, protected_body, max_distance=margin + 0.02)
        if d < -0.005:
            print(f"[WARN] Contact detected ({d:.4f} m); stopping arm motion")
            break


def move_arm_via_conf_safe(env, target_conf, via_conf, protected_body=None,
                           margin=0.05, steps=12, sub_steps=10):
    """Two-segment arm motion with clearance margin on both segments."""
    move_arm_to_conf_safe(env, via_conf, protected_body=protected_body,
                          margin=margin, steps=steps, sub_steps=sub_steps)
    move_arm_to_conf_safe(env, target_conf, protected_body=protected_body,
                          margin=margin, steps=steps, sub_steps=sub_steps)


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
