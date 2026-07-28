# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Replanning TAMP controller for RoboEXP.

Provides a closed-loop execution layer: observe -> plan -> execute -> re-observe
-> replan if the world diverges from the planner's model.  The observer is
abstract so it can be backed by perception (``PerceptionObserver``) or by direct
PyBullet state (``GTStateObserver``).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import numpy as np

from pybullet_tools.utils import get_pose
from pybullet_tools.pr2_utils import get_arm_joints, get_joint_positions
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import print_solution

from roboexp.env import pr2_skills
from roboexp.tamp.perception_tamp_bridge import PerceptionTAMPBridge
from roboexp.tamp.problem_builder import (
    TAMPProblemBuilder,
    _compute_on_relationships,
    _is_movable,
    _is_surface,
)


DEFAULT_POS_TOL = 0.05
DEFAULT_ORN_TOL = 0.2
DEFAULT_MAX_REPLANS = 3


@dataclass
class StateSnapshot:
    """Planning-relevant snapshot of the PyBullet world.

    The object pose representation is the same hashable tuple that
    ``pybullet_tools.utils.get_pose`` returns: ``((x, y, z), (qx, qy, qz, qw))``.
    """

    robot_base_conf: Tuple[float, float, float]
    robot_arm_conf: Tuple[float, ...]
    holding_body: Optional[int]
    object_poses: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    on_relationships: Set[Tuple[int, int]] = field(default_factory=set)


class Observer(ABC):
    """Abstract state source for the replanning controller."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable observer name for logging."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, env) -> StateSnapshot:
        """Return a fresh planning-relevant state snapshot."""
        raise NotImplementedError


class GTStateObserver(Observer):
    """Read the planning-relevant state directly from PyBullet.

    This is useful for validating the closed-loop logic without requiring the
    visual perception pipeline (which may fail on unsupported hardware or when
    the camera viewpoint is poor).  A ``PerceptionObserver`` can be swapped in
    later without changing the controller.
    """

    def __init__(
        self,
        movable_categories: Optional[Set[str]] = None,
        surface_categories: Optional[Set[str]] = None,
    ):
        self.movable_categories = set(movable_categories or set())
        self.surface_categories = set(surface_categories or set())

    def name(self) -> str:
        return "GTStateObserver"

    def observe(self, env) -> StateSnapshot:
        world = env.world
        robot_body = env.robot.body

        base_conf = self._get_base_conf(robot_body)
        arm_joints = get_arm_joints(robot_body, "left")
        arm_conf = tuple(float(v) for v in get_joint_positions(robot_body, arm_joints))
        holding_body = getattr(env, "grasped_body", None)

        # Track all planning-relevant bodies (movables + surfaces).
        movables = []
        surfaces = []
        tracked_bodies = set()
        for body, obj in world.BODY_TO_OBJECT.items():
            if body == robot_body:
                continue
            if _is_surface(obj, self.surface_categories):
                surfaces.append(body)
                tracked_bodies.add(body)
            elif _is_movable(obj, self.movable_categories):
                movables.append(body)
                tracked_bodies.add(body)

        object_poses = {b: tuple(get_pose(b)) for b in tracked_bodies}
        on_relationships = set(_compute_on_relationships(world, movables, surfaces))

        return StateSnapshot(
            robot_base_conf=base_conf,
            robot_arm_conf=arm_conf,
            holding_body=holding_body,
            object_poses=object_poses,
            on_relationships=on_relationships,
        )

    @staticmethod
    def _get_base_conf(robot_body):
        from pybullet_tools.utils import get_pose, euler_from_quat

        point, quat = get_pose(robot_body)
        x, y = float(point[0]), float(point[1])
        roll, pitch, yaw = euler_from_quat(quat)
        return (x, y, float(yaw))


class PerceptionObserver(Observer):
    """Observe through the visual perception pipeline and fall back to world state.

    The visual pipeline is responsible for detecting the movable object and the
    support/target surfaces.  If any of them is missed (e.g. the target surface is
    outside the camera view), the observer falls back to ``world.BODY_TO_OBJECT``
    category/name matching so the controller can still run.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        movable_label: str = "cup",
        support_label: str = "counter",
        target_label: str = "table",
        device: str = "cuda",
    ):
        self.labels = labels or ["table", "counter", "cup"]
        self.movable_label = movable_label
        self.support_label = support_label
        self.target_label = target_label
        self.device = device
        self._bridge: Optional[PerceptionTAMPBridge] = None

    def name(self) -> str:
        return "PerceptionObserver"

    def observe(self, env) -> StateSnapshot:
        if self._bridge is None:
            self._bridge = PerceptionTAMPBridge(
                env,
                object_level_labels=self.labels,
                device=self.device,
            )
        self._bridge.observe_and_build_graph()
        entities = self._bridge.resolve_tamp_entities(
            movable_label=self.movable_label,
            support_surface_label=self.support_label,
            target_surface_label=self.target_label,
        )

        world = env.world
        robot_body = env.robot.body

        base_conf = GTStateObserver._get_base_conf(robot_body)
        arm_joints = get_arm_joints(robot_body, "left")
        arm_conf = tuple(float(v) for v in get_joint_positions(robot_body, arm_joints))
        holding_body = getattr(env, "grasped_body", None)

        movable_body = entities.get("movable_body") or self._find_body_by_label(
            world, self.movable_label
        )
        support_surface_body = entities.get(
            "support_surface_body"
        ) or self._find_body_by_label(world, self.support_label)
        target_surface_body = entities.get(
            "target_surface_body"
        ) or self._find_body_by_label(world, self.target_label)

        tracked_bodies = {
            b
            for b in (movable_body, support_surface_body, target_surface_body)
            if b is not None
        }
        object_poses = {b: tuple(get_pose(b)) for b in tracked_bodies}

        on_relationships = set()
        if movable_body is not None and support_surface_body is not None:
            # Re-use the same on-test as the problem builder.
            on_pairs = _compute_on_relationships(
                world, [movable_body], [support_surface_body]
            )
            on_relationships.update(on_pairs)
        if movable_body is not None and target_surface_body is not None:
            on_pairs = _compute_on_relationships(
                world, [movable_body], [target_surface_body]
            )
            on_relationships.update(on_pairs)

        return StateSnapshot(
            robot_base_conf=base_conf,
            robot_arm_conf=arm_conf,
            holding_body=holding_body,
            object_poses=object_poses,
            on_relationships=on_relationships,
        )

    @staticmethod
    def _find_body_by_label(world, label: str) -> Optional[int]:
        label = label.lower()
        for body, obj in world.BODY_TO_OBJECT.items():
            if obj.category.lower() == label:
                return body
            if label in obj.name.lower():
                return body
        return None


def _pose_close(
    p1: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    p2: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    pos_tol: float = DEFAULT_POS_TOL,
    _orn_tol: float = DEFAULT_ORN_TOL,
) -> bool:
    """Whether two poses are close enough to be considered unchanged.

    For the small upright objects used in this domain, position drift is the
    dominant signal; orientation is ignored because the objects are kept upright.
    """
    pos1 = np.array(p1[0], dtype=float)
    pos2 = np.array(p2[0], dtype=float)
    return float(np.linalg.norm(pos1 - pos2)) <= pos_tol


def _action_expected_holding_and_on(prev: StateSnapshot, action) -> Tuple[Optional[int], Set[Tuple[int, int]]]:
    """Return the expected (holding_body, on_relationships) after ``action``."""
    if hasattr(action, "name"):
        name = action.name
        args = action.args
    else:
        name = action[0]
        args = action[1:]

    if name == "pick" and len(args) >= 2:
        obj = args[1]
        support = None
        for o, s in prev.on_relationships:
            if o == obj:
                support = s
                break
        expected_holding = obj
        expected_on = {pair for pair in prev.on_relationships if pair[0] != obj}
        return expected_holding, expected_on

    if name == "place" and len(args) >= 7:
        obj = args[1]
        surface = args[6]
        expected_holding = None
        expected_on = set(prev.on_relationships)
        expected_on.add((obj, surface))
        return expected_holding, expected_on

    # move_base / move_arm / any other action: world should be unchanged except
    # for the attached object moving with the robot (which we ignore).
    return prev.holding_body, set(prev.on_relationships)


def detect_unexpected_changes(
    prev: StateSnapshot,
    curr: StateSnapshot,
    action,
    pos_tol: float = DEFAULT_POS_TOL,
    orn_tol: float = DEFAULT_ORN_TOL,
) -> List[str]:
    """Return a list of unexpected state changes after executing ``action``.

    The function is action-aware: ``pick`` and ``place`` are expected to change
    the held object and its support relation, while ``move_base``/``move_arm``
    are expected to leave the world unchanged except for the attached object
    moving with the robot.
    """
    if hasattr(action, "name"):
        name = action.name
        args = action.args
    else:
        name = action[0]
        args = action[1:]

    involved_obj = None
    if name in ("pick", "place") and len(args) >= 2:
        involved_obj = args[1]

    expected_holding, expected_on = _action_expected_holding_and_on(prev, action)
    changes: List[str] = []

    if curr.holding_body != expected_holding:
        changes.append(
            f"holding_body: expected {expected_holding}, got {curr.holding_body}"
        )

    if curr.on_relationships != expected_on:
        changes.append(
            f"on_relationships: expected {sorted(expected_on)}, got {sorted(curr.on_relationships)}"
        )

    # Pose changes.
    all_bodies = set(prev.object_poses.keys()) | set(curr.object_poses.keys())
    for body in all_bodies:
        in_prev = body in prev.object_poses
        in_curr = body in curr.object_poses
        if not in_prev:
            changes.append(f"body {body} appeared in observation")
            continue
        if not in_curr:
            changes.append(f"body {body} disappeared from observation")
            continue

        held_prev = prev.holding_body == body
        held_curr = curr.holding_body == body
        if held_prev and held_curr:
            # The object is attached to the robot; its world pose is expected to
            # change during base/arm motions, so we ignore it.
            continue

        if body == involved_obj and name in ("pick", "place"):
            # The object involved in a pick/place is expected to move.
            continue

        if not _pose_close(prev.object_poses[body], curr.object_poses[body], pos_tol, orn_tol):
            changes.append(f"pose of body {body} changed")

    return changes


def default_goal_reached(
    snapshot: StateSnapshot,
    goal: Tuple,
) -> bool:
    """Default goal test for ``("On" ?obj ?surface)`` goals."""
    if isinstance(goal, (list, tuple)) and len(goal) == 3 and goal[0] in ("On", "on"):
        obj = goal[1]
        surface = goal[2]
        return (obj, surface) in snapshot.on_relationships
    return False


class ReplanningTAMPController:
    """Closed-loop TAMP controller with action-aware replanning.

    Parameters
    ----------
    env : PyBulletExplorationEnv
        The simulation environment.
    observer : Observer
        State source (``GTStateObserver`` or ``PerceptionObserver``).
    problem_builder : TAMPProblemBuilder
        Builder that constructs a PDDLStream problem from the current world.
    max_replans : int
        Maximum number of times to replan after detecting unexpected change.
    pos_tol, orn_tol : float
        Tolerances used when comparing object poses.
    """

    def __init__(
        self,
        env,
        observer: Observer,
        problem_builder: TAMPProblemBuilder,
        max_replans: int = DEFAULT_MAX_REPLANS,
        pos_tol: float = DEFAULT_POS_TOL,
        orn_tol: float = DEFAULT_ORN_TOL,
    ):
        self.env = env
        self.observer = observer
        self.builder = problem_builder
        self.max_replans = max_replans
        self.pos_tol = pos_tol
        self.orn_tol = orn_tol

    def _solve_problem(self, problem, world_view):
        import pddlstream.algorithms.visualization as viz
        viz.set_visualizations_false()
        solution = solve(
            problem,
            algorithm="adaptive",
            unit_costs=True,
            debug=False,
            world=world_view,
        )
        if solution is not None and solution[0] is not None:
            print_solution(solution)
        return solution

    def run(
        self,
        goal: Tuple,
        execute_action_fn: Callable[[Any, Any], None],
        goal_reached_fn: Optional[Callable[[StateSnapshot, Tuple], bool]] = None,
        on_replan_fn: Optional[Callable[[int, Any, List[str]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the observe-plan-execute-replan loop.

        Parameters
        ----------
        goal : tuple
            PDDL goal expression, e.g. ``("On", cup_body, table_body)``.
        execute_action_fn : callable
            ``execute_action_fn(env, action)`` executes one PDDLStream action.
        goal_reached_fn : callable, optional
            ``goal_reached_fn(snapshot, goal) -> bool``.
        on_replan_fn : callable, optional
            ``on_replan_fn(replan_idx, action_or_none, changes)`` called whenever
            a replan is triggered.

        Returns
        -------
        dict with keys ``status``, ``replans``, ``total_actions``,
        ``final_plan``, and ``reason`` (on failure).
        """
        if goal_reached_fn is None:
            goal_reached_fn = default_goal_reached

        world_view = _WorldView(self.env.world)
        cup_body = goal[1]
        target_surface_body = goal[2]
        support_surface_body: Optional[int] = None

        total_actions_executed = 0
        executed_plan: List[Any] = []

        for replan_idx in range(self.max_replans + 1):
            snapshot = self.observer.observe(self.env)
            print(f"\n=== Replan {replan_idx} | observer={self.observer.name()} ===")
            print(f"  holding={snapshot.holding_body}, on={sorted(snapshot.on_relationships)}")

            if goal_reached_fn(snapshot, goal):
                print("Goal already reached.")
                return {
                    "status": "success",
                    "replans": replan_idx,
                    "total_actions": total_actions_executed,
                    "final_plan": executed_plan,
                }

            # Derive / update the support surface for the movable object.
            for b, s in snapshot.on_relationships:
                if b == cup_body:
                    support_surface_body = s
                    break

            problem = self.builder.build_problem(
                goal,
                holding_body=snapshot.holding_body,
                support_surface=support_surface_body,
                target_surface=target_surface_body,
                relevant_movables=[cup_body],
                relevant_surfaces=[
                    b
                    for b in (support_surface_body, target_surface_body)
                    if b is not None
                ],
            )
            solution = self._solve_problem(problem, world_view)
            if solution is None or solution[0] is None:
                print(f"Planning failed at replan {replan_idx}")
                return {
                    "status": "failure",
                    "replans": replan_idx,
                    "total_actions": total_actions_executed,
                    "final_plan": executed_plan,
                    "reason": "no_plan",
                }

            plan, cost, _evaluations = solution
            print(f"Plan ({len(plan)} actions): {[a.name for a in plan]}")

            executed_this_iteration: List[Any] = []
            for action in plan:
                if goal_reached_fn(self.observer.observe(self.env), goal):
                    print("Goal reached before next action.")
                    return {
                        "status": "success",
                        "replans": replan_idx,
                        "total_actions": total_actions_executed,
                        "final_plan": executed_plan,
                    }

                execute_action_fn(self.env, action)
                total_actions_executed += 1
                executed_this_iteration.append(action)
                executed_plan.append(action)

                new_snapshot = self.observer.observe(self.env)
                if goal_reached_fn(new_snapshot, goal):
                    print("Goal reached after action.")
                    return {
                        "status": "success",
                        "replans": replan_idx,
                        "total_actions": total_actions_executed,
                        "final_plan": executed_plan,
                    }

                changes = detect_unexpected_changes(
                    snapshot,
                    new_snapshot,
                    action,
                    self.pos_tol,
                    self.orn_tol,
                )
                if changes:
                    print(f"Unexpected changes after {action.name}: {changes}")
                    if on_replan_fn is not None:
                        on_replan_fn(replan_idx, action, changes)
                    snapshot = new_snapshot
                    break

                snapshot = new_snapshot
            else:
                # Inner loop completed without break: all actions in this plan
                # executed and no unexpected changes detected.  If the goal is not
                # reached, the model is incomplete; replan from the current state.
                if goal_reached_fn(self.observer.observe(self.env), goal):
                    return {
                        "status": "success",
                        "replans": replan_idx,
                        "total_actions": total_actions_executed,
                        "final_plan": executed_plan,
                    }
                print("Plan executed but goal not reached; replanning from current state.")
                if on_replan_fn is not None:
                    on_replan_fn(replan_idx, None, ["goal not reached after plan"])
                continue

        return {
            "status": "failure",
            "replans": self.max_replans,
            "total_actions": total_actions_executed,
            "final_plan": executed_plan,
            "reason": "max_replans_reached",
        }


class _WorldView:
    """Thin wrapper so PDDLStream's debug printer gets string names."""

    def __init__(self, world):
        self.world = world

    def get_debug_name(self, value):
        if isinstance(value, int):
            return str(self.world.get_debug_name(value))
        return str(value)
