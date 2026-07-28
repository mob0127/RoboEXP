# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Perception-TAMP bridge for RoboEXP.

This module connects RoboEXP's perception pipeline (RoboPercept + RoboMemory +
ActionSceneGraph) to the native TAMP problem builder.  It captures a head-camera
observation, runs detection/segmentation, fuses the result into a scene graph,
and maps scene-graph nodes back to PyBullet bodies so that TAMP can plan over
perceived rather than hand-specified entities.
"""
import numpy as np

from pybullet_tools.utils import get_aabb, get_aabb_center

from roboexp.perception.robo_percept import RoboPercept
from roboexp.memory.robo_memory import RoboMemory


# Object-level labels that the perception stack should look for in the kitchen
# scene.  Keeping the list short reduces false positives and avoids unrelated
# scene clutter from entering the TAMP problem.
DEFAULT_OBJECT_LEVEL_LABELS = ["table", "counter", "cup"]

DEFAULT_GROUNDING_DICT = " . ".join(DEFAULT_OBJECT_LEVEL_LABELS) + " ."

# Workspace bounds for the voxel memory.  These should cover the reachable
# kitchen area.
DEFAULT_LOWER_BOUND = [-2, -2, -0.5]
DEFAULT_HIGHER_BOUND = [6, 6, 3]
DEFAULT_VOXEL_SIZE = 0.02

# Viewpoint used for the single-shot perception phase.  It is chosen so that
# the PR2 head camera has a clear view of the counter and the cup placed on it.
DEFAULT_PERCEPTION_VIEWPOINT = {
    "base": (0.30, 0.00, 0.00),
    "head": (0.00, 0.50),
    "torso": 0.25,
}


class PerceptionTAMPBridge:
    """
    Run RoboEXP perception and expose the resulting scene graph to TAMP.

    Parameters
    ----------
    env : PyBulletExplorationEnv
        The environment containing the PyBullet world and robot.
    object_level_labels : list[str], optional
        Labels passed to RoboMemory as object-level entities.
    device : str, optional
        Device passed to RoboPercept (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        env,
        object_level_labels=None,
        device="cuda",
    ):
        self.env = env
        self.world = env.world
        self.object_level_labels = object_level_labels or DEFAULT_OBJECT_LEVEL_LABELS
        self.grounding_dict = " . ".join(self.object_level_labels) + " ."
        self.device = device

        self.percept = RoboPercept(
            grounding_dict=self.grounding_dict,
            lazy_loading=False,
            device=device,
        )
        self.memory = RoboMemory(
            lower_bound=DEFAULT_LOWER_BOUND,
            higher_bound=DEFAULT_HIGHER_BOUND,
            voxel_size=DEFAULT_VOXEL_SIZE,
            real_camera=False,
        )

    # ------------------------------------------------------------------
    # Observation & scene-graph construction
    # ------------------------------------------------------------------
    def observe_and_build_graph(self, viewpoint=None):
        """
        Move the robot to a perception viewpoint, capture observations, run
        RoboPercept, update RoboMemory, and build/return the ActionSceneGraph.

        Parameters
        ----------
        viewpoint : dict, optional
            Override the default perception viewpoint.  Keys: ``base``,
            ``head``, ``torso``.

        Returns
        -------
        ActionSceneGraph
        """
        import pybullet as p
        from pybullet_tools.pr2_utils import set_group_conf
        from roboexp.env import pr2_skills

        viewpoint = viewpoint or DEFAULT_PERCEPTION_VIEWPOINT
        base_xytheta = viewpoint["base"]
        head_conf = viewpoint["head"]
        torso_value = viewpoint.get("torso", 0.25)

        # Move to perception viewpoint.
        pr2_skills.set_pr2_root(self.env, *base_xytheta)
        set_group_conf(self.env.robot.body, "torso", [torso_value])
        set_group_conf(self.env.robot.body, "head", head_conf)
        for _ in range(60):
            p.stepSimulation()

        observations = self.env.get_observations(wrist_only=True)
        attributes = self.percept.get_attributes_from_observations(
            observations, visualize=False
        )

        self.memory.update_memory(
            observations,
            attributes,
            self.object_level_labels,
            filter_masks={},
            update_scene_graph=True,
            scene_graph_option=None,
            visualize=False,
        )

        return self.memory.action_scene_graph

    # ------------------------------------------------------------------
    # Scene graph → body mapping
    # ------------------------------------------------------------------
    def get_scene_graph(self):
        """Return the current ActionSceneGraph, if any."""
        return self.memory.action_scene_graph

    def find_nodes_by_label(self, label):
        """Return all scene-graph object nodes whose label matches ``label``."""
        sg = self.memory.action_scene_graph
        if sg is None:
            return []
        return [
            node
            for node in sg.object_nodes.values()
            if node.node_label == label
        ]

    def map_node_to_body(
        self,
        node,
        candidate_bodies=None,
        category_hint=None,
    ):
        """
        Map a scene-graph node to the best-matching PyBullet body.

        Matching is done by nearest 3D AABB-center distance.  If
        ``candidate_bodies`` is provided, only those bodies are considered;
        otherwise all bodies in ``world.BODY_TO_OBJECT`` are considered.
        If ``category_hint`` is provided, bodies whose category does not match
        are skipped.

        Returns
        -------
        int or None
            The matched body id, or None if no candidate is within tolerance.
        """
        if getattr(node.instance, "voxel_indexes", None) is None:
            # Scene-graph nodes without geometry (e.g. the synthetic table root)
            # cannot be matched to bodies.
            return None
        node_center = np.array(node.instance.get_attributes()["center"])

        if candidate_bodies is None:
            candidate_bodies = list(self.world.BODY_TO_OBJECT.keys())

        best_body = None
        best_dist = float("inf")
        for body in candidate_bodies:
            if not isinstance(body, int):
                continue
            obj = self.world.BODY_TO_OBJECT.get(body)
            if obj is None:
                continue
            if category_hint is not None and not self._category_matches(obj, category_hint):
                continue
            body_center = np.array(get_aabb_center(get_aabb(body)))
            dist = np.linalg.norm(node_center - body_center)
            if dist < best_dist:
                best_dist = dist
                best_body = body

        # Tolerance: scene-graph nodes can be noisy; reject clearly wrong matches.
        # Perception in this kitchen setup can offset small objects by ~0.6 m
        # vertically, so keep a permissive threshold.
        if best_dist > 2.5:
            return None
        return best_body

    def resolve_tamp_entities(
        self,
        movable_label="cup",
        target_surface_label="table",
        support_surface_label="counter",
    ):
        """
        Resolve the main TAMP entities from the perceived scene graph.

        Returns
        -------
        dict with keys ``movable_body``, ``target_surface_body``,
        ``support_surface_body``.
        """
        movable_body = self._resolve_body_by_label(
            movable_label, category_hint=movable_label
        )
        target_surface_body = self._resolve_body_by_label(
            target_surface_label, category_hint=target_surface_label
        )
        support_surface_body = self._resolve_body_by_label(
            support_surface_label, category_hint=support_surface_label
        )
        return {
            "movable_body": movable_body,
            "target_surface_body": target_surface_body,
            "support_surface_body": support_surface_body,
        }

    def _category_matches(self, obj, hint):
        """Flexible category matching that maps natural labels to world categories."""
        if obj.category == hint:
            return True
        if hint in ("counter", "table") and obj.category == "supporter":
            return True
        if hint in obj.name.lower():
            return True
        return False

    def _resolve_body_by_label(self, label, category_hint=None):
        """Find the first scene-graph node with ``label`` and map it to a body."""
        # When the world already has a uniquely named object for this label
        # (e.g. the fixed ``counter`` in the kitchen scene), prefer it over
        # noisy perception-based matching.
        if category_hint is not None:
            named_obj = self.world.name_to_object(category_hint)
            if named_obj is not None and self._category_matches(
                named_obj, category_hint
            ):
                return named_obj.body

        nodes = self.find_nodes_by_label(label)
        if not nodes:
            return None
        # If multiple nodes match, prefer the one closest to a body of the
        # hinted category.
        if category_hint is not None:
            candidate_bodies = [
                b for b, obj in self.world.BODY_TO_OBJECT.items()
                if isinstance(b, int) and self._category_matches(obj, category_hint)
            ]
        else:
            candidate_bodies = None

        best_body = None
        best_dist = float("inf")
        for node in nodes:
            if getattr(node.instance, "voxel_indexes", None) is None:
                continue
            body = self.map_node_to_body(node, candidate_bodies=candidate_bodies)
            if body is not None:
                node_center = np.array(node.instance.get_attributes()["center"])
                body_center = np.array(get_aabb_center(get_aabb(body)))
                dist = np.linalg.norm(node_center - body_center)
                if dist < best_dist:
                    best_dist = dist
                    best_body = body
        return best_body
