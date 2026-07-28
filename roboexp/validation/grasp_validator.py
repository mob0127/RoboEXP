# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Grasp outcome validator for the PyBullet-based RoboEXP pipeline.

This module classifies a pick/grasp attempt into one of three states:

- success: the object is physically attached to the gripper and has been lifted
  while remaining upright.
- missed: the gripper closed but the object was not grasped (still on its
  support surface).
- tipped: the object was moved but is now in an abnormal orientation or has
  fallen over.

NOTE: This validator is tied to PyBullet ground-truth poses and is therefore a
simulation-only utility.  It is not applicable to a real robot.
"""
import numpy as np
import pybullet as p

from pybullet_tools.utils import get_pose, get_aabb, point_from_pose


def _quat_rotate(quat, vec):
    """Rotate a vector by a quaternion (x, y, z, w)."""
    x, y, z, w = quat
    vx, vy, vz = vec
    # q * v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ])


class GraspValidator:
    """
    Validate a pick/grasp attempt by comparing the object's PyBullet state
    before and after the action.

    Parameters
    ----------
    env : PyBulletExplorationEnv
        The environment that (optionally) performs physical grasping.
    up_axis : tuple or ndarray
        World up direction (default: +Z).
    lift_threshold : float
        Minimum increase in the object's bottom-z to count as "lifted" (m).
    tip_threshold : float
        Minimum dot product between the object's local up vector and world up
        to count as "upright".  Values below this are considered tipped.
    """

    def __init__(self, env, up_axis=(0, 0, 1), lift_threshold=0.03,
                 tip_threshold=0.7, displacement_threshold=0.10):
        self.env = env
        self.up_axis = np.array(up_axis, dtype=np.float32)
        self.up_axis = self.up_axis / (np.linalg.norm(self.up_axis) + 1e-8)
        self.lift_threshold = lift_threshold
        self.tip_threshold = tip_threshold
        self.displacement_threshold = displacement_threshold
        self._snapshots = {}

    # ------------------------------------------------------------------
    # Body resolution
    # ------------------------------------------------------------------
    def _resolve_body(self, target):
        """Map a target to a PyBullet body id.

        Supported targets:
          - int / body id
          - any object with a ``.body`` attribute (e.g. ``world_builder.Object``)
          - ``myInstance`` (uses label + nearest-center matching)
          - ``ObjectNode`` (uses its ``.instance``)
        """
        if isinstance(target, int):
            return target
        if hasattr(target, "body") and isinstance(target.body, int):
            return target.body

        instance = None
        if hasattr(target, "instance"):
            instance = target.instance
        elif hasattr(target, "label") and hasattr(target, "voxel_indexes"):
            instance = target

        if instance is None:
            raise ValueError(f"Cannot resolve target {target!r} to a PyBullet body")

        return self._resolve_instance_body(instance)

    def _resolve_instance_body(self, instance):
        """Find the world object whose category and 3D center best match the instance."""
        label = getattr(instance, "label", None)
        if label is None:
            raise ValueError("Instance has no label")

        instance_center = np.mean(
            instance.index_to_pcd(instance.voxel_indexes), axis=0
        )

        # Candidate objects: prefer category match, fall back to all objects.
        candidates = list(self.env.world.OBJECTS_BY_CATEGORY.get(label, []))
        if not candidates:
            candidates = list(self.env.world.BODY_TO_OBJECT.values())

        best_body, best_dist = None, float("inf")
        for obj in candidates:
            body = getattr(obj, "body", None)
            if body is None or body == self.env.robot.body:
                continue
            aabb = get_aabb(body)
            center = (np.array(aabb[0]) + np.array(aabb[1])) / 2.0
            dist = np.linalg.norm(center - instance_center)
            if dist < best_dist:
                best_dist, best_body = dist, body

        if best_body is None:
            raise RuntimeError(f"No matching body found for instance label '{label}'")
        return best_body

    # ------------------------------------------------------------------
    # Snapshot / classification
    # ------------------------------------------------------------------
    def snapshot(self, target):
        """Record the current physical state of ``target`` as the pre-grasp reference."""
        body = self._resolve_body(target)
        pose = get_pose(body)
        aabb = get_aabb(body)
        position = np.array(point_from_pose(pose))
        up_vector = _quat_rotate(pose[1], (0, 0, 1))

        snap = {
            "body": body,
            "position": position,
            "aabb": (np.array(aabb[0]), np.array(aabb[1])),
            "support_z": aabb[0][2],
            "up_vector": up_vector,
        }
        self._snapshots[body] = snap
        return snap

    def _get_state(self, body):
        pose = get_pose(body)
        aabb = get_aabb(body)
        position = np.array(point_from_pose(pose))
        up_vector = _quat_rotate(pose[1], (0, 0, 1))
        return {
            "position": position,
            "aabb": (np.array(aabb[0]), np.array(aabb[1])),
            "support_z": aabb[0][2],
            "up_vector": up_vector,
        }

    def classify(self, target):
        """Classify the grasp outcome relative to the last snapshot.

        Returns one of: ``'success'``, ``'missed'``, ``'tipped'``.
        """
        body = self._resolve_body(target)
        if body not in self._snapshots:
            raise RuntimeError("No snapshot recorded for target; call snapshot() first")

        before = self._snapshots[body]
        after = self._get_state(body)

        # Is the object physically attached to the gripper?
        attached = (
            getattr(self.env, "physical_grasp", False)
            and getattr(self.env, "grasped_body", None) == body
        )

        delta_z = after["support_z"] - before["support_z"]
        up_dot = float(np.dot(after["up_vector"], self.up_axis))
        horizontal_disp = float(np.linalg.norm(
            (after["position"] - before["position"])[:2]
        ))

        if attached and delta_z > self.lift_threshold and up_dot > self.tip_threshold:
            return "success"

        # Tipped/knocked over if it lost upright pose, fell, or was displaced
        # significantly while not being lifted.
        if up_dot < self.tip_threshold or delta_z < -self.lift_threshold:
            return "tipped"
        if not attached and horizontal_disp > self.displacement_threshold:
            return "tipped"

        return "missed"

    def get_report(self, target):
        """Return a human-readable dict with before/after metrics."""
        body = self._resolve_body(target)
        before = self._snapshots[body]
        after = self._get_state(body)
        attached = (
            getattr(self.env, "physical_grasp", False)
            and getattr(self.env, "grasped_body", None) == body
        )
        return {
            "body": body,
            "attached": attached,
            "before_position": before["position"].tolist(),
            "after_position": after["position"].tolist(),
            "before_support_z": float(before["support_z"]),
            "after_support_z": float(after["support_z"]),
            "before_up_dot": float(np.dot(before["up_vector"], self.up_axis)),
            "after_up_dot": float(np.dot(after["up_vector"], self.up_axis)),
            "outcome": self.classify(target),
        }
