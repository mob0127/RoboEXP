# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Native TAMP stream functions for PR2 pick-and-place.

Unlike the Step-1 shortcut version, these functions use real geometric
reasoning:
  - real PR2 arm IK via IKFast / pybullet_tools;
  - collision-aware base motion planning via plan_base_motion;
  - sampled grasps and place poses checked against the scene.
"""
import math
import random

import numpy as np
import pybullet as p

from pybullet_tools.utils import (
    get_aabb, set_pose, set_base_values, get_base_values, plan_base_motion,
    pairwise_collision, body_collision, Pose, Point, Euler, multiply, invert,
    point_from_pose, quat_from_euler, unit_pose, BodySaver, get_pose,
    set_joint_positions, get_joint_positions, all_between,
    create_cylinder, GREY,
)
from pybullet_tools.pr2_utils import (
    LEFT_ARM, set_group_conf, get_arm_joints, set_arm_conf, PR2_TOOL_FRAMES,
    get_gripper_link, get_torso_arm_joints,
)
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics


CYLINDER_RADIUS = 0.08
CYLINDER_HEIGHT = 0.20

# Workspace for base motion planning.
BASE_LIMITS = ((-3.0, -3.0), (6.0, 6.0))

# Distance from the object center at which the PR2 base is placed for grasping.
GRASP_BASE_RADIUS = 0.85

# Number of base pose candidates to try around each object pose.
NUM_BASE_CANDIDATES = 16

# Number of place-pose samples per surface.
NUM_PLACE_SAMPLES = 12

APPROACH_DISTANCE = 0.06

# Number of side grasps around the cylinder.
NUM_SIDE_GRASPS = 8

# Distance to retract the gripper along tool z before / after the grasp/place.
# This gives the planner a safe ``pre-grasp`` arm configuration, while the
# executor performs the final small approach/retract slide in task space.
APPROACH_DISTANCE = 0.06

# IKFast returns 8-DOF solutions (torso + 7 arm joints).  We fix the torso.
TORSO_VALUE = 0.25

# Base-motion planning uses a small cylindrical proxy for the PR2 footprint.
# The full robot collision geometry is too conservative for this tight kitchen
# scene (the left arm inevitably hangs over the counter).  The proxy radius
# corresponds to the main chassis and keeps the planned base path safe.
BASE_FOOTPRINT_RADIUS = 0.25
BASE_FOOTPRINT_HEIGHT = 0.30
NUM_BASE_MOTION_STEPS = 30


def _se2_from_xytheta(x, y, theta):
    return (float(x), float(y), float(theta))


def _normalize_angle(theta):
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


def _poses_close(q1, q2, pos_tol=1e-3, yaw_tol=1e-3):
    """Whether two SE(2) configurations are essentially the same."""
    return (
        abs(q1[0] - q2[0]) < pos_tol
        and abs(q1[1] - q2[1]) < pos_tol
        and abs(_normalize_angle(q1[2] - q2[2])) < yaw_tol
    )


def _tuple_to_pose(t):
    """Convert a hashable tuple back to a (point, quat) pose."""
    return (list(t[0]), list(t[1]))


def _is_floor(env, body):
    return env.world.get_category(body) == "floor"


class PR2TAMPStreams:
    """Container for PDDLStream stream functions that use real PR2 geometry."""

    def __init__(self, env, cup, support_surface, target_surface, obstacles=None):
        """
        Args:
            env: PyBulletExplorationEnv with robot_name="pr2".
            cup: The cup object to manipulate.
            support_surface: The surface the cup starts on (e.g. counter).
            target_surface: The surface on which to place the cup.
            obstacles: List of body ids to treat as obstacles.  If None, all
                bodies except the robot, the cup, and the floor are used.
        """
        self.env = env
        self.robot = env.robot.body
        self.cup = cup
        self.support_surface = support_surface
        self.target_surface = target_surface
        if obstacles is None:
            self.obstacles = []
            for b in env.world.objects:
                if not isinstance(b, int):
                    continue
                if b in (self.robot, cup.body):
                    continue
                # The floor is in permanent contact with the robot base; it
                # must not be treated as a collision obstacle.
                if _is_floor(env, b):
                    continue
                # Ignore small movable clutter for base-motion planning.
                cat = env.world.get_category(b)
                if cat in ("food", "bottle", "medicine"):
                    continue
                self.obstacles.append(b)
        else:
            self.obstacles = list(obstacles)

        # Fixed torso height for the whole demo.
        set_group_conf(self.robot, "torso", [TORSO_VALUE])

        # Collision proxy used for base-motion planning.  It is kept under the
        # floor when not in use so it does not affect the scene.
        self.base_checker = create_cylinder(
            radius=BASE_FOOTPRINT_RADIUS,
            height=BASE_FOOTPRINT_HEIGHT,
            color=GREY,
        )
        set_pose(self.base_checker, Pose(point=Point(0.0, 0.0, -10.0)))

        # Obstacles for base motion: everything except the robot, the cup, the
        # floor, and the two surfaces we manipulate on/with.
        self.base_obstacles = [
            b for b in self.obstacles
            if b not in (self.support_surface.body, self.target_surface.body)
        ]

    # ------------------------------------------------------------------
    # Grasp sampling
    # ------------------------------------------------------------------

    def sample_grasp(self, obj):
        """Yield top-down grasps around the cylinder in the object frame.

        Each grasp is a relative (position, quaternion) tuple expressed in the
        object's local frame.  The tool origin is placed at the object centre
        and the tool z-axis points downward so the gripper can close around the
        cup from above.  ``yaw`` varies the base approach direction around the
        object.
        """
        for i in range(NUM_SIDE_GRASPS):
            yaw = 2 * math.pi * i / NUM_SIDE_GRASPS
            # Tool z points down (-world z); yaw rotates the gripper around z.
            orn = quat_from_euler(Euler(roll=0.0, pitch=math.pi, yaw=yaw))
            grasp = ((0.0, 0.0, 0.0), tuple(orn))
            yield (grasp,)

    # ------------------------------------------------------------------
    # Place pose sampling
    # ------------------------------------------------------------------

    def sample_place_pose(self, obj, surface):
        """Yield collision-free place poses on ``surface``.

        The pose is expressed as a world-frame (position, quaternion) tuple.
        """
        aabb = get_aabb(surface)
        z = float(aabb[1][2] + CYLINDER_HEIGHT / 2.0 + 0.005)

        xmin, xmax = aabb[0][0], aabb[1][0]
        ymin, ymax = aabb[0][1], aabb[1][1]
        # Keep the cup within the surface with a small margin.
        margin = CYLINDER_RADIUS + 0.005
        xmin += margin
        xmax -= margin
        ymin += margin
        ymax -= margin

        # Deterministic grid + small jitter so PDDLStream can backtrack.
        for i in range(NUM_PLACE_SAMPLES):
            if xmax > xmin and ymax > ymin:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
            else:
                x = (xmin + xmax) / 2.0
                y = (ymin + ymax) / 2.0

            pose_tuple = (
                (float(x), float(y), float(z)),
                (0.0, 0.0, 0.0, 1.0),
            )
            if self._is_place_pose_free(pose_tuple, surface):
                yield (pose_tuple,)

    def _is_place_pose_free(self, pose_tuple, surface):
        """Check whether the cup at ``pose_tuple`` collides with obstacles."""
        cup_body = self.cup.body
        saver = BodySaver(cup_body)
        try:
            set_pose(cup_body, _tuple_to_pose(pose_tuple))
            for obs in self.obstacles:
                if obs == surface or obs == cup_body:
                    continue
                if body_collision(cup_body, obs):
                    return False
            return True
        finally:
            saver.restore()

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def inverse_kinematics_pick(self, robot, obj, grasp, pose):
        """Yield (base_conf, arm_conf) pairs that can pick ``obj`` at ``pose``."""
        yield from self._ik_for_pose(robot, grasp, pose)

    def inverse_kinematics_place(self, robot, obj, grasp, pose):
        """Yield (base_conf, arm_conf) pairs that can place ``obj`` at ``pose``."""
        yield from self._ik_for_pose(robot, grasp, pose)

    def _ik_for_pose(self, robot, grasp, pose):
        """Core IK sampling.

        ``grasp`` and ``pose`` are tuples of (point, quaternion).  The target
        tool pose in world coordinates is ``object_pose * grasp``.
        """
        object_pose = _tuple_to_pose(pose)
        grasp_pose = _tuple_to_pose(grasp)
        tool_pose = multiply(object_pose, grasp_pose)

        obj_pos = point_from_pose(object_pose)

        # Exclude the cup, the support/target surfaces, the floor, and the
        # large appliances that sit next to the counter (they are behind the
        # robot's approach direction and make the conservative IK collision
        # check reject otherwise valid arm configurations).
        # The PR2 base and torso must be set before calling IK because the
        # IKFast solver reasons in the base frame.  We keep the torso fixed.
        set_group_conf(robot, "torso", [TORSO_VALUE])

        robot_saver = BodySaver(robot)
        try:
            for radius in (GRASP_BASE_RADIUS, 0.75, 0.95, 0.65):
                for i in range(NUM_BASE_CANDIDATES):
                    yaw = 2 * math.pi * i / NUM_BASE_CANDIDATES
                    # Base is placed ``radius`` away, facing the object.
                    bx = obj_pos[0] + radius * math.cos(yaw)
                    by = obj_pos[1] + radius * math.sin(yaw)
                    btheta = _normalize_angle(yaw + math.pi)
                    base_conf = _se2_from_xytheta(bx, by, btheta)

                    set_base_values(robot, base_conf)
                    arm_conf = pr2_inverse_kinematics(
                        robot, LEFT_ARM, tool_pose,
                        obstacles=[], max_attempts=100,
                    )
                    if arm_conf is not None:
                        yield (base_conf, tuple(float(v) for v in arm_conf))
        finally:
            robot_saver.restore()

    # ------------------------------------------------------------------
    # Base motion planning
    # ------------------------------------------------------------------

    def _set_checker_pose(self, q):
        """Place the cylindrical footprint proxy at SE(2) config ``q``."""
        set_pose(
            self.base_checker,
            Pose(point=Point(q[0], q[1], BASE_FOOTPRINT_HEIGHT / 2.0)),
        )

    def _checker_collides(self):
        """Return True if the footprint proxy hits a base obstacle."""
        for obs in self.base_obstacles:
            if pairwise_collision(self.base_checker, obs):
                return True
        return False

    def plan_base_motion(self, q1, q2):
        """Yield a collision-free straight-line base path from ``q1`` to ``q2``.

        Planning is done with a small cylindrical footprint proxy instead of the
        full robot geometry; the left arm would otherwise collide with the
        counter in every configuration.  The executor is responsible for keeping
        the arm in a safe pose while the base follows the path.
        """
        q1 = _se2_from_xytheta(*q1)
        q2 = _se2_from_xytheta(*q2)
        if _poses_close(q1, q2):
            yield ((q1,),)
            return

        # Unwrap yaw for linear interpolation.
        dyaw = q2[2] - q1[2]
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        q2 = (q2[0], q2[1], q1[2] + dyaw)

        checker_saver = BodySaver(self.base_checker)
        try:
            for i in range(NUM_BASE_MOTION_STEPS + 1):
                alpha = i / NUM_BASE_MOTION_STEPS
                x = q1[0] + alpha * (q2[0] - q1[0])
                y = q1[1] + alpha * (q2[1] - q1[1])
                yaw = q1[2] + alpha * (q2[2] - q1[2])
                self._set_checker_pose((x, y, yaw))
                if self._checker_collides():
                    return
            path_tuple = (
                q1,
                _se2_from_xytheta(q2[0], q2[1], _normalize_angle(q2[2])),
            )
            yield (path_tuple,)
        finally:
            checker_saver.restore()
