# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
TAMPProblemBuilder: build PDDLStream problems from a PyBullet world state.

This module scans the world for robots, movable objects, and surfaces, extracts
current robot base/arm configurations and object poses, and synthesizes the
initial facts and goal for the PR2 pick-and-place domain.
"""
import os
import sys

import numpy as np

from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_gen_fn
from pddlstream.utils import read

from pybullet_tools.utils import get_aabb, get_pose, get_mass, get_bodies
from pybullet_tools.pr2_utils import (
    get_arm_joints, get_joint_positions, LEFT_ARM, RIGHT_ARM,
)

from roboexp.tamp.pr2_stream_functions import PR2TAMPStreams


DEFAULT_MOVABLE_CATEGORIES = {
    "cup", "bottle", "can", "bowl", "mug", "glass", "spoon", "fork",
    "knife", "apple", "banana", "orange", "object", "movable",
}

DEFAULT_SURFACE_CATEGORIES = {
    "counter", "table", "supporter", "surface", "sink", "stove",
    "microwave", "oven", "minifridge", "fridge", "shelf",
}

# Categories that are never planning entities.
DEFAULT_IGNORED_CATEGORIES = {
    "floor", "robot", "camera", "marker", "debug",
}


def _get_body_category(world, body):
    """Return the category string registered for a PyBullet body, if any."""
    if body in world.BODY_TO_OBJECT:
        return world.BODY_TO_OBJECT[body].category
    return None


def _get_body_name(world, body):
    """Return the name registered for a PyBullet body, if any."""
    if body in world.BODY_TO_OBJECT:
        return world.BODY_TO_OBJECT[body].name
    if body in world.ROBOT_TO_OBJECT:
        return world.ROBOT_TO_OBJECT[body].name
    return None


def _is_surface(obj, surface_categories=None):
    """Heuristic: a surface is a large, mostly-static body with a flat top."""
    if surface_categories is None:
        surface_categories = DEFAULT_SURFACE_CATEGORIES
    cat = obj.category.lower()
    if cat in surface_categories or cat in DEFAULT_SURFACE_CATEGORIES:
        return True
    # Also trust explicit supported_objects annotations from world builders.
    if getattr(obj, "supported_objects", None):
        return True
    return False


def _is_movable(obj, movable_categories=None):
    """Heuristic: a movable object is small, has mass, and is not a surface/robot."""
    if movable_categories is None:
        movable_categories = DEFAULT_MOVABLE_CATEGORIES
    cat = obj.category.lower()
    if cat in DEFAULT_IGNORED_CATEGORIES:
        return False
    if cat in movable_categories or cat in DEFAULT_MOVABLE_CATEGORIES:
        return True
    # Fallback: small mass and not a surface.
    try:
        mass = get_mass(obj.body)
    except Exception:
        mass = 0.0
    if mass <= 0.0:
        return False
    return not _is_surface(obj)


def _compute_on_relationships(world, movable_bodies, surface_bodies, threshold=0.02):
    """
    Determine which movable body is resting on which surface body.

    A body is considered ``on`` a surface if:
      * its bottom z is within ``threshold`` of the surface top z, and
      * its horizontal footprint overlaps the surface's footprint.
    """
    on_pairs = []
    for body in movable_bodies:
        body_aabb = get_aabb(body)
        body_bottom = body_aabb[0][2]
        body_xy = [(body_aabb[0][0], body_aabb[0][1]), (body_aabb[1][0], body_aabb[1][1])]

        best_surface = None
        best_top = -np.inf
        for surface in surface_bodies:
            surf_aabb = get_aabb(surface)
            surf_top = surf_aabb[1][2]
            if abs(body_bottom - surf_top) > threshold:
                continue
            # Horizontal overlap test.
            if (
                body_xy[0][0] < surf_aabb[1][0] and body_xy[1][0] > surf_aabb[0][0]
                and body_xy[0][1] < surf_aabb[1][1] and body_xy[1][1] > surf_aabb[0][1]
            ):
                if surf_top > best_top:
                    best_top = surf_top
                    best_surface = surface
        if best_surface is not None:
            on_pairs.append((body, best_surface))
    return on_pairs


def _resolve_body(world, spec):
    """Resolve a body specifier (int body id or string name) to a body id."""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        body = world.name_to_body(spec)
        if body is None:
            raise ValueError(f"No body named '{spec}' in world.")
        return body
    # Some callers pass Object instances.
    if hasattr(spec, "body"):
        return spec.body
    raise ValueError(f"Cannot resolve body specifier: {spec!r}")


class TAMPProblemBuilder:
    """
    Build a PDDLStream problem for PR2 pick-and-place from a PyBullet world.

    Parameters
    ----------
    env : PyBulletExplorationEnv
        The environment wrapping the PyBullet ``World`` and robot.
    arm : str
        Which arm to plan for (``"left"`` or ``"right"``).
    movable_categories : set, optional
        Extra category names to treat as movable objects.
    surface_categories : set, optional
        Extra category names to treat as surfaces.
    """

    def __init__(
        self,
        env,
        arm="left",
        movable_categories=None,
        surface_categories=None,
    ):
        self.env = env
        self.world = env.world
        self.robot_body = env.robot.body
        self.arm = arm
        self.arm_joints = get_arm_joints(self.robot_body, self.arm)

        self.movable_categories = set(movable_categories or set())
        self.surface_categories = set(surface_categories or set())

    # ------------------------------------------------------------------
    # World scanning
    # ------------------------------------------------------------------
    def scan_objects(self):
        """
        Scan the world and classify bodies into robots, movables, and surfaces.

        Returns
        -------
        dict with keys ``robot``, ``movables``, ``surfaces``.
        """
        movables = []
        surfaces = []

        for body, obj in self.world.BODY_TO_OBJECT.items():
            if body == self.robot_body:
                continue
            if _is_surface(obj, self.surface_categories):
                surfaces.append(body)
            elif _is_movable(obj, self.movable_categories):
                movables.append(body)

        return {
            "robot": self.robot_body,
            "movables": movables,
            "surfaces": surfaces,
        }

    def get_current_base_conf(self):
        """Return the current robot base configuration as (x, y, theta).

        Reads the base's world pose directly (``get_pose``) rather than the
        ``base`` joint group, because some setup helpers such as
        ``set_pr2_root`` reset the world pose without updating the joint values.
        """
        from pybullet_tools.utils import get_pose, euler_from_quat
        point, quat = get_pose(self.robot_body)
        x, y = float(point[0]), float(point[1])
        roll, pitch, yaw = euler_from_quat(quat)
        return (x, y, float(yaw))

    def get_current_arm_conf(self):
        """Return the current arm joint configuration as a tuple."""
        return tuple(float(v) for v in get_joint_positions(self.robot_body, self.arm_joints))

    def get_body_pose(self, body):
        """Return the current pose of ``body`` as PDDLStream expects: ((x,y,z), (qx,qy,qz,qw))."""
        return tuple(get_pose(body))

    # ------------------------------------------------------------------
    # PDDL problem generation
    # ------------------------------------------------------------------
    def build_problem(
        self,
        goal,
        holding_body=None,
        support_surface=None,
        target_surface=None,
        relevant_movables=None,
        relevant_surfaces=None,
    ):
        """
        Build a PDDLStream problem from the current world state.

        Parameters
        ----------
        goal : tuple
            A PDDL goal expression, e.g. ``("On", "cup", "target_table")``.
            Object names are resolved against the world; body ids are used as-is.
        holding_body : int or str, optional
            If the robot is already holding an object, pass it here.  The
            initial ``(Holding ?r ?o)`` fact will be emitted instead of
            ``(HandEmpty ?r)``.
        support_surface : int or str, optional
            The surface the held object was picked from.  Required only if
            ``holding_body`` is given (for the ``KinPlace`` stream).
        target_surface : int or str, optional
            The surface to place on.  If omitted, inferred from the goal.
        relevant_movables : list, optional
            Explicit list of movable bodies (or names) to include in the
            problem.  If provided, the builder will only treat these bodies as
            movable objects, which avoids PDDLStream binding conflicts caused
            by unrelated scene clutter.  If omitted, the builder infers them
            from the goal and the current support relationships.
        relevant_surfaces : list, optional
            Explicit list of surface bodies (or names) to include in the
            problem.  If omitted, inferred from the goal and the current
            support relationships.

        Returns
        -------
        PDDLProblem
        """
        scanned = self.scan_objects()
        robot = scanned["robot"]

        # Resolve goal names once; this is used both for relevance inference
        # and for picking the support/target surfaces later.
        resolved_goal = self._resolve_goal(goal)

        # Compute support relationships over all candidate objects first, so
        # we can infer which bodies are actually relevant to the goal.
        all_on_pairs = _compute_on_relationships(
            self.world, scanned["movables"], scanned["surfaces"]
        )

        if relevant_movables is not None or relevant_surfaces is not None:
            # User provided explicit lists; use them directly.
            movables = (
                [_resolve_body(self.world, b) for b in relevant_movables]
                if relevant_movables is not None
                else scanned["movables"]
            )
            surfaces = (
                [_resolve_body(self.world, b) for b in relevant_surfaces]
                if relevant_surfaces is not None
                else scanned["surfaces"]
            )
        else:
            # Infer relevant entities from the goal to avoid PDDLStream
            # binding conflicts caused by unrelated scene clutter.
            movables, surfaces = self._infer_relevant_entities(
                resolved_goal, scanned, all_on_pairs
            )
            if not movables:
                raise ValueError(
                    "Could not infer any relevant movable objects from the goal. "
                    "Try passing relevant_movables explicitly."
                )

        base_q0 = self.get_current_base_conf()
        arm_q0 = self.get_current_arm_conf()

        # Load PDDL files relative to this module.
        tamp_dir = os.path.join(os.path.dirname(__file__))
        domain_pddl = read(os.path.join(tamp_dir, "domains", "pr2_pick_place_domain.pddl"))
        stream_pddl = read(os.path.join(tamp_dir, "streams", "pr2_pick_place_stream.pddl"))

        init = [
            ("Robot", robot),
            ("BConf", base_q0),
            ("AConf", arm_q0),
            ("RobotAt", robot, base_q0),
            ("ArmAt", robot, arm_q0),
        ]

        for body in movables:
            init.append(("IsObject", body))
        for body in surfaces:
            init.append(("Surface", body))

        # Object poses and On relationships.
        on_pairs = _compute_on_relationships(self.world, movables, surfaces)
        placed = set()
        for body, surface in on_pairs:
            pose = self.get_body_pose(body)
            init.append(("Pose", pose))
            init.append(("ObjectAt", body, pose))
            init.append(("On", body, surface))
            placed.add(body)

        # Any movable not resting on a known surface still gets a Pose/ObjectAt.
        for body in movables:
            if body not in placed:
                pose = self.get_body_pose(body)
                init.append(("Pose", pose))
                init.append(("ObjectAt", body, pose))

        # Hand state.
        if holding_body is not None:
            held_body = _resolve_body(self.world, holding_body)
            init.append(("Holding", robot, held_body))
        else:
            init.append(("HandEmpty", robot))

        # Determine the target surface for stream construction.
        if target_surface is None:
            target_surface = self._infer_target_surface(resolved_goal)
        else:
            target_surface = _resolve_body(self.world, target_surface)

        if target_surface is None:
            raise ValueError("Could not infer a target surface from the goal; please pass target_surface.")

        # Determine the support surface (where the moved object starts).
        if support_surface is None:
            support_surface = self._infer_support_surface(resolved_goal, on_pairs)
        else:
            support_surface = _resolve_body(self.world, support_surface)

        if support_surface is None:
            raise ValueError("Could not infer the object's initial support surface; please pass support_surface.")

        # Find the primary movable object involved in the goal.
        goal_movable = self._infer_goal_movable(resolved_goal, movables)

        # PR2TAMPStreams expects Object instances, not raw body ids.
        cup_obj = self.world.BODY_TO_OBJECT[goal_movable]
        support_obj = self.world.BODY_TO_OBJECT[support_surface]
        target_obj = self.world.BODY_TO_OBJECT[target_surface]

        streams = PR2TAMPStreams(self.env, cup_obj, support_obj, target_obj)
        stream_map = {
            "sample-grasp": from_gen_fn(streams.sample_grasp),
            "sample-place-pose": from_gen_fn(streams.sample_place_pose),
            "inverse-kinematics-pick": from_gen_fn(streams.inverse_kinematics_pick),
            "inverse-kinematics-place": from_gen_fn(streams.inverse_kinematics_place),
            "plan-base-motion": from_gen_fn(streams.plan_base_motion),
        }

        return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, resolved_goal)

    # ------------------------------------------------------------------
    # Goal / object resolution helpers
    # ------------------------------------------------------------------
    def _resolve_goal(self, goal):
        """Recursively resolve string object names inside a PDDL goal expression to body ids.

        Predicate names (strings that do not map to a body) are left unchanged.
        """
        if isinstance(goal, str):
            try:
                return _resolve_body(self.world, goal)
            except ValueError:
                return goal
        if isinstance(goal, (list, tuple)):
            return tuple(self._resolve_goal(g) for g in goal)
        return goal

    def _infer_target_surface(self, goal):
        """Try to pull the target surface out of a simple ``On`` goal."""
        if isinstance(goal, (list, tuple)) and len(goal) == 3 and goal[0] in ("On", "on"):
            return goal[2]
        return None

    def _infer_support_surface(self, goal, on_pairs):
        """Try to find the surface the goal object currently rests on."""
        if isinstance(goal, (list, tuple)) and len(goal) == 3 and goal[0] in ("On", "on"):
            obj = goal[1]
            for body, surface in on_pairs:
                if body == obj:
                    return surface
        return None

    def _infer_goal_movable(self, goal, movables):
        """Return the movable body mentioned in the goal, if any."""
        if isinstance(goal, (list, tuple)) and len(goal) == 3 and goal[0] in ("On", "on"):
            obj = goal[1]
            if obj in movables:
                return obj
        if movables:
            return movables[0]
        raise ValueError("No movable objects found in the world; cannot build TAMP problem.")

    def _infer_relevant_entities(self, goal, scanned, on_pairs):
        """
        Infer the set of movable objects and surfaces relevant to ``goal``.

        For a simple ``(On ?o ?s)`` goal, the relevant entities are:
          * the goal object ``?o``
          * the goal surface ``?s``
          * the surface that currently supports ``?o`` (so the initial ``On``
            fact can be stated)

        This filters out unrelated scene clutter (other bottles, walls,
        appliances) that would otherwise blow up the PDDLStream search space.
        """
        movables = set()
        surfaces = set()

        goal_movable = None
        goal_surface = None
        if isinstance(goal, (list, tuple)) and len(goal) == 3 and goal[0] in ("On", "on"):
            goal_movable = goal[1]
            goal_surface = goal[2]

        if goal_movable is not None:
            movables.add(goal_movable)
        if goal_surface is not None:
            surfaces.add(goal_surface)

        # Add the current support surface of the goal movable, if any.
        if goal_movable is not None:
            for body, surface in on_pairs:
                if body == goal_movable:
                    surfaces.add(surface)
                    break

        return list(movables), list(surfaces)
