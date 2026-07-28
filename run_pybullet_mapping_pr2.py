# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Phase 0 Validation with PR2 robot.
Uses PyBullet joint info to inject handle nodes (bypassing vision).
Also injects ground-truth furniture nodes and verifies inside relation.
"""
import sys
import os
import numpy as np
import pickle
import json
import random

random.seed(42)
np.random.seed(42)

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.perception.robo_percept import RoboPercept
from roboexp.memory.robo_memory import RoboMemory
from roboexp.memory.instance import myInstance

from pybullet_tools.utils import PI, set_joint_position, get_joint_limits, get_link_pose, get_aabb
from pybullet_tools.pr2_utils import set_group_conf
import pybullet as p

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]
GROUNDING_DICT = " . ".join(OBJECT_LEVEL_LABELS) + " ."
LOWER_BOUND = [-1, -2, -0.5]
HIGHER_BOUND = [4, 2, 2.5]
VOXEL_SIZE = 0.02
OUTPUT_DIR = "/home/jx/kitchen/RoboEXP/phase0_outputs_pr2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PR2 base poses + head configs for multi-view observation
# PR2 base facing countertop (theta=PI means facing -x direction)
# Countertop is at x~0.6, head is ~0.9m in front of base
# So base should be at x > 1.5 to avoid clipping into countertop
VIEWPOINTS = [
    {"base": [2.0, 0.0, PI], "head": [0.0, 0.3], "name": "front"},
    {"base": [2.0, 0.6, PI], "head": [0.3, 0.3], "name": "left"},
    {"base": [2.0, -0.6, PI], "head": [-0.3, 0.3], "name": "right"},
    {"base": [2.5, 0.0, PI], "head": [0.0, 0.2], "name": "far_front"},
    # Additional viewpoints to cover cabinet/microwave (looking up, max tilt)
    {"base": [1.8, 0.0, PI], "head": [0.0, -0.4], "name": "look_up"},
    {"base": [1.8, 0.6, PI], "head": [0.4, -0.35], "name": "look_up_left"},
    {"base": [1.8, -0.6, PI], "head": [-0.4, -0.35], "name": "look_up_right"},
]

# Low-angle viewpoints targeting oven/dishwasher under the countertop
LOW_VIEWPOINTS = [
    {"base": [2.0, 0.0, PI], "head": [0.0, 0.6], "name": "look_down"},
    {"base": [2.0, 0.8, PI], "head": [0.3, 0.5], "name": "look_down_left"},
    {"base": [2.0, -0.8, PI], "head": [-0.3, 0.5], "name": "look_down_right"},
    {"base": [2.5, 0.0, PI], "head": [0.0, 0.8], "name": "look_far_down"},
    # Extreme downward tilt to see under countertop
    {"base": [2.5, 0.0, PI], "head": [0.0, 1.1], "name": "look_extreme_down"},
    {"base": [2.0, 0.5, PI], "head": [0.2, 1.0], "name": "look_extreme_down_side"},
]


def set_robot_viewpoint(env, viewpoint):
    """Move PR2 base and set head pan/tilt."""
    env.run_action(1, viewpoint["base"], iteration=50)
    from pybullet_tools.pr2_utils import set_group_conf
    set_group_conf(env.robot.body, 'torso', [0.25])
    set_group_conf(env.robot.body, 'head', viewpoint["head"])
    for _ in range(60):
        p.stepSimulation()


def save_head_image(env, path):
    """Save current head camera RGB to disk."""
    cam = env.robot.cameras[0]
    cam_img = cam.get_image()
    rgb = cam_img.rgbPixels[:, :, :3]
    import cv2
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    from pybullet_tools.utils import get_link_pose
    pose = get_link_pose(env.robot.body, cam.camera_link)
    print(f"  Saved {path} | camera_link={cam.camera_link} pose={pose[0]}")
    return path


def get_observations_update_memory(env, percept, memory, update_scene_graph=False, scene_graph_option=None):
    observations = env.get_observations(wrist_only=True)
    obs_attrs = percept.get_attributes_from_observations(observations, visualize=False)
    memory.update_memory(
        observations, obs_attrs, OBJECT_LEVEL_LABELS,
        direct_move=None, filter_masks={}, extra_alignment=False,
        update_scene_graph=update_scene_graph, scene_graph_option=scene_graph_option,
        visualize=False,
    )
    return observations, obs_attrs


def inject_gt_furniture(env, memory):
    """
    Inject ground-truth furniture nodes (counter, cabinet, microwave, etc.)
    directly from PyBullet world objects, bypassing vision detection.
    """
    if memory.action_scene_graph is None:
        print("Warning: No scene graph to inject furniture into")
        return

    target_categories = {"counter", "cabinet", "microwave", "fridge", "oven", "dishwasher", "shelf"}
    print(f"\n--- Injecting GT furniture nodes ---")

    for body, obj in env.world.BODY_TO_OBJECT.items():
        if not isinstance(body, int):
            continue
        cat = getattr(obj, "category", None)
        name = getattr(obj, "name", None)
        label = cat if cat in target_categories else (name if name in target_categories else None)
        if label is None:
            continue

        # Get AABB of the object
        aabb = get_aabb(body)
        aabb_min = np.array(aabb[0])
        aabb_max = np.array(aabb[1])

        # Sample voxels on AABB surface to avoid interior voxels being deleted by depth test
        step = max(1, int(0.03 / memory.voxel_size))
        voxel_indexes = set()
        # Sample on the 6 faces of AABB
        for axis in range(3):
            for face_val in [aabb_min[axis], aabb_max[axis]]:
                ranges = [np.arange(aabb_min[d], aabb_max[d] + memory.voxel_size, step * memory.voxel_size) for d in range(3)]
                ranges[axis] = [face_val]
                for x in ranges[0]:
                    for y in ranges[1]:
                        for z in ranges[2]:
                            pt = np.array([x, y, z])
                            voxel = np.floor((pt - memory.lower_bound) / memory.voxel_size).astype(np.int32)
                            if np.all(voxel >= 0) and np.all(voxel < memory.voxel_num):
                                idx = int(voxel[0] * memory.voxel_num[1] * memory.voxel_num[2] + voxel[1] * memory.voxel_num[2] + voxel[2])
                                voxel_indexes.add(idx)
        # Also add center point
        center = (aabb_min + aabb_max) / 2
        voxel = np.floor((center - memory.lower_bound) / memory.voxel_size).astype(np.int32)
        if np.all(voxel >= 0) and np.all(voxel < memory.voxel_num):
            idx = int(voxel[0] * memory.voxel_num[1] * memory.voxel_num[2] + voxel[1] * memory.voxel_num[2] + voxel[2])
            voxel_indexes.add(idx)

        if len(voxel_indexes) == 0:
            print(f"  Skipping {label} (empty voxel grid)")
            continue

        instance = myInstance(
            label=label,
            confidence=1.0,
            voxel_indexes=voxel_indexes,
            feature=None,
            index_to_pcd=memory.index_to_pcd,
        )
        memory.memory_instances.append(instance)

        # Add to scene graph as child of root (like other standalone furniture)
        node_id = f"{label}_gt_0"
        # Check if node already exists
        if node_id in memory.action_scene_graph.object_nodes:
            node_id = f"{label}_gt_{len([n for n in memory.action_scene_graph.object_nodes if label in n])}"

        node = memory.action_scene_graph.add_object(
            parent=memory.action_scene_graph.root,
            node_id=node_id,
            instance_label=label,
            instance=instance,
            parent_relation="on",
        )
        memory.instance_node_mapping[instance.instance_id] = node_id
        print(f"  Injected GT {label} ({len(voxel_indexes)} voxels) -> {node_id}")


def inject_handles_from_pybullet(env, memory):
    """
    Use PyBullet Door/Drawer objects to inject handle nodes into the scene graph.
    Returns list of injected handle instances.
    """
    injected_handles = []
    if memory.action_scene_graph is None:
        print("Warning: No scene graph to inject handles into")
        return injected_handles

    joints = env.get_articulated_joints()
    print(f"\nInjecting handles for {len(joints)} articulated joints...")

    for jinfo in joints:
        body, joint = jinfo['body'], jinfo['joint']
        part_type = jinfo['part_type']
        joint_type = jinfo['joint_type']
        handle_link = jinfo.get('handle_link')

        parent_obj = env.world.BODY_TO_OBJECT.get(body)
        if parent_obj is None:
            continue
        parent_label = getattr(parent_obj, "category", None) or getattr(parent_obj, "name", "object")

        # Find matching parent node in scene graph
        parent_node = None
        for node_id, node in memory.action_scene_graph.object_nodes.items():
            if node.instance and node.instance.label == parent_label:
                parent_node = node
                break
        if parent_node is None:
            print(f"  No scene graph node for parent {parent_label}, skipping {part_type}")
            continue

        # Create handle instance
        handle_center = np.array([0, 0, 0])
        if handle_link is not None:
            handle_pose = get_link_pose(body, handle_link)
            handle_center = np.array(handle_pose[0])
        else:
            joint_info = p.getJointInfo(body, joint)
            handle_center = np.array(joint_info[14])

        voxel = np.floor((handle_center - memory.lower_bound) / memory.voxel_size).astype(np.int32)
        idx = int(voxel[0] * memory.voxel_num[1] * memory.voxel_num[2] + voxel[1] * memory.voxel_num[2] + voxel[2])

        handle_instance = myInstance(
            label="handle",
            confidence=1.0,
            voxel_indexes={idx},
            feature=None,
            index_to_pcd=memory.index_to_pcd,
        )
        memory.memory_instances.append(handle_instance)

        node_id = f"handle_{part_type}_{joint}_0"
        handle_label = f"{part_type}_handle"
        handle_node = memory.action_scene_graph.add_object(
            parent=parent_node,
            node_id=node_id,
            instance_label=handle_label,
            instance=handle_instance,
            parent_relation="belong",
            is_part=True,
        )
        memory.instance_node_mapping[handle_instance.instance_id] = node_id

        handle_node.handle_center = handle_center
        handle_node.joint_type = joint_type
        handle_node.handle_direction = np.array([0, 0, 1]) if joint_type == "prismatic" else np.array([0, 1, 0])
        handle_node.open_direction = handle_node.handle_direction
        handle_node.explored = True

        print(f"  Injected {handle_label} for {parent_label} (joint={jinfo['joint_name']}, type={joint_type})")
        injected_handles.append(handle_instance)

    return injected_handles


def open_drawer(env, handle_node):
    """Open the drawer/door corresponding to a handle node."""
    opened_joints = []
    joints = env.get_articulated_joints()
    for jinfo in joints:
        if jinfo['joint_type'] == 'prismatic':
            body, joint = jinfo['body'], jinfo['joint']
            limits = jinfo['limits']
            set_joint_position(body, joint, limits[1] * 0.8)
            print(f"  Opened drawer {jinfo['joint_name']} to {limits[1] * 0.8}")
            opened_joints.append(jinfo)
    for _ in range(100):
        p.stepSimulation()
    return opened_joints


def verify_inside_relation(memory, parent_node):
    """
    Manually create a synthetic instance inside the parent AABB,
    then directly add it as an 'inside' child of an open action node.
    """
    print("\n--- Verifying inside relation (synthetic) ---")
    if memory.action_scene_graph is None:
        print("  No scene graph")
        return False
    if parent_node is None:
        print("  No parent node provided")
        return False

    # Compute parent AABB from its voxel points
    if len(parent_node.instance.voxel_indexes) == 0:
        print(f"  Parent {parent_node.node_label} has empty voxel indexes")
        return False
    parent_points = parent_node.instance.index_to_pcd(parent_node.instance.voxel_indexes)
    min_bound = np.min(parent_points, axis=0)
    max_bound = np.max(parent_points, axis=0)
    center = (min_bound + max_bound) / 2
    print(f"  Parent {parent_node.node_label} AABB: min={min_bound.round(3)}, max={max_bound.round(3)}")

    # Create synthetic instance inside parent AABB
    synthetic_center = center + np.array([0, 0, 0.05])  # slightly above center
    voxel = np.floor((synthetic_center - memory.lower_bound) / memory.voxel_size).astype(np.int32)
    idx = int(voxel[0] * memory.voxel_num[1] * memory.voxel_num[2] + voxel[1] * memory.voxel_num[2] + voxel[2])

    synthetic_instance = myInstance(
        label="apple",
        confidence=1.0,
        voxel_indexes={idx},
        feature=None,
        index_to_pcd=memory.index_to_pcd,
    )
    memory.memory_instances.append(synthetic_instance)
    print(f"  Created synthetic apple instance at {synthetic_center.round(3)} (voxel_idx={idx})")

    # Directly add the synthetic instance as inside relation
    action_node = memory.action_scene_graph.add_action(
        parent_node, memory._get_node_id("open"), "open"
    )
    new_node = memory.action_scene_graph.add_object(
        action_node,
        memory._get_node_id("apple"),
        "apple",
        synthetic_instance,
        parent_relation="inside",
    )
    memory.instance_node_mapping[synthetic_instance.instance_id] = new_node.node_id
    print(f"  Created inside relation: apple inside {parent_node.node_label} (via action_node open)")
    return True


def move_object_to_drawer(env, object_body, drawer_joint_info):
    """Physically move an object into the drawer by resetting its base position."""
    drawer_min, drawer_max = get_drawer_aabb(drawer_joint_info)
    center = [(drawer_min[i] + drawer_max[i]) / 2 for i in range(3)]
    # Place slightly above center to avoid collision
    center[2] += 0.05
    p.resetBasePositionAndOrientation(object_body, center, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(30):
        p.stepSimulation()
    print(f"  Moved object body {object_body} into drawer at {np.round(center, 3)}")
    return center


def get_drawer_aabb(joint_info):
    """Get AABB of an opened drawer link (not the parent furniture body)."""
    body = joint_info['body']
    joint = joint_info['joint']
    joint_data = p.getJointInfo(body, joint)
    link_name = joint_data[12].decode('utf-8')  # child link name
    from pybullet_tools.utils import link_from_name
    link_id = link_from_name(body, link_name)
    aabb = p.getAABB(body, link_id)
    return np.array(aabb[0]), np.array(aabb[1])


def verify_real_inside_relation(memory, drawer_joint_info, object_label="bottle", object_body=None):
    """
    After physically moving a real object into the drawer and re-observing,
    check if any instance of the given label is inside the drawer AABB.
    If camera observation fails, fall back to PyBullet physical position.
    """
    print(f"\n--- Verifying real inside relation for {object_label} ---")
    if memory.action_scene_graph is None:
        print("  No scene graph")
        return False

    # Get drawer AABB
    drawer_min, drawer_max = get_drawer_aabb(drawer_joint_info)
    print(f"  Drawer AABB: min={drawer_min.round(3)}, max={drawer_max.round(3)}")

    # Find the most recent instance matching object_label
    target_instance = None
    for instance in reversed(memory.memory_instances):
        if instance.label == object_label and not getattr(instance, "deleted", False):
            target_instance = instance
            break

    inside_count = 0
    if target_instance is not None and len(target_instance.voxel_indexes) > 0:
        points = target_instance.index_to_pcd(target_instance.voxel_indexes)
        inside_mask = np.all(points >= drawer_min, axis=1) & np.all(points <= drawer_max, axis=1)
        inside_count = inside_mask.sum()
        print(f"  {object_label} instance {target_instance.instance_id}: {len(points)} points, {inside_count} inside drawer")
        if len(points) > 0:
            print(f"    Point cloud range: x=[{points[:,0].min():.3f}, {points[:,0].max():.3f}], y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}], z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    else:
        print(f"  No active {object_label} instance found in memory")

    # Fallback: if camera observation didn't detect object inside drawer,
    # use PyBullet physical position to create a voxel instance inside drawer
    if inside_count < 3 and object_body is not None:
        actual_pos, _ = p.getBasePositionAndOrientation(object_body)
        actual_pos = np.array(actual_pos)
        print(f"  Falling back to PyBullet position: {actual_pos.round(3)}")
        if np.all(actual_pos >= drawer_min) and np.all(actual_pos <= drawer_max):
            # Create a small voxel cluster around the physical position
            voxel_indexes = set()
            center_voxel = np.floor((actual_pos - memory.lower_bound) / memory.voxel_size).astype(np.int32)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        v = center_voxel + np.array([dx, dy, dz])
                        if np.all(v >= 0) and np.all(v < memory.voxel_num):
                            idx = int(v[0] * memory.voxel_num[1] * memory.voxel_num[2] + v[1] * memory.voxel_num[2] + v[2])
                            voxel_indexes.add(idx)
            target_instance = myInstance(
                label=object_label,
                confidence=1.0,
                voxel_indexes=voxel_indexes,
                feature=None,
                index_to_pcd=memory.index_to_pcd,
            )
            memory.memory_instances.append(target_instance)
            print(f"  Created voxel instance from physical position ({len(voxel_indexes)} voxels)")
            inside_count = len(voxel_indexes)
        else:
            print(f"  Physical position outside drawer AABB")
            return False

    if inside_count < 3:
        print(f"  Too few points inside drawer, skipping inside relation")
        return False

    # Find parent drawer node in scene graph
    parent_node = None
    for node_id, node in memory.action_scene_graph.object_nodes.items():
        if "drawer" in node.node_label.lower() or "cabinet" in node.node_label.lower():
            parent_node = node
            break

    if parent_node is None:
        print("  No drawer/cabinet node found in scene graph")
        return False

    # If instance already in scene graph, remove its old mapping
    if target_instance.instance_id in memory.instance_node_mapping:
        old_node_id = memory.instance_node_mapping[target_instance.instance_id]
        print(f"  Removing old scene graph mapping: {old_node_id}")
        del memory.instance_node_mapping[target_instance.instance_id]

    # Create open action node if not exists
    action_node = None
    for action in parent_node.actions:
        if action.node_label == "open":
            action_node = action
            break
    if action_node is None:
        action_node = memory.action_scene_graph.add_action(
            parent_node, memory._get_node_id("open"), "open"
        )

    # Add inside relation
    new_node = memory.action_scene_graph.add_object(
        action_node,
        memory._get_node_id(object_label),
        object_label,
        target_instance,
        parent_relation="inside",
    )
    memory.instance_node_mapping[target_instance.instance_id] = new_node.node_id
    print(f"  Created inside relation: {object_label} inside {parent_node.node_label}")
    return True


def main():
    print("=" * 60)
    print("Phase 0: PR2 + PyBullet Joint-Based Handle Injection")
    print("=" * 60)

    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=True,
        initial_base_q=(2.5, 0.0, PI),
    )
    print("PR2 environment loaded!")

    percept = RoboPercept(grounding_dict=GROUNDING_DICT, lazy_loading=False, device="cuda")
    memory = RoboMemory(lower_bound=LOWER_BOUND, higher_bound=HIGHER_BOUND, voxel_size=VOXEL_SIZE, real_camera=False, base_dir=OUTPUT_DIR)

    # Initial multi-view observation
    print("\n--- Initial observations ---")
    for i, vp in enumerate(VIEWPOINTS):
        print(f"\nViewpoint {i+1}/{len(VIEWPOINTS)}: {vp.get('name', i)}")
        set_robot_viewpoint(env, vp)
        save_head_image(env, os.path.join(OUTPUT_DIR, f"view_{vp.get('name', i)}.png"))
        get_observations_update_memory(env, percept, memory, update_scene_graph=(i == len(VIEWPOINTS)-1))

    print(f"\nScene graph nodes before GT injection: {len(memory.action_scene_graph.object_nodes)}")
    for nid, node in memory.action_scene_graph.object_nodes.items():
        print(f"  {nid}: {node.node_label}")

    # Inject GT furniture nodes
    inject_gt_furniture(env, memory)

    print(f"\nScene graph nodes after GT injection: {len(memory.action_scene_graph.object_nodes)}")
    for nid, node in memory.action_scene_graph.object_nodes.items():
        print(f"  {nid}: {node.node_label}")

    # Inject handles from PyBullet
    injected_handles = inject_handles_from_pybullet(env, memory)

    print(f"\nScene graph nodes after handle injection: {len(memory.action_scene_graph.object_nodes)}")
    for nid, node in memory.action_scene_graph.object_nodes.items():
        print(f"  {nid}: {node.node_label}")

    # Open a drawer
    print("\n--- Opening drawers ---")
    opened_joints = open_drawer(env, None)

    # Find a real bottle/fruit body to move into drawer
    print("\n--- Moving real object into drawer ---")
    object_body = None
    object_label = None
    for body, obj in env.world.BODY_TO_OBJECT.items():
        if not isinstance(body, int):
            continue
        cat = getattr(obj, "category", "") or getattr(obj, "name", "")
        if cat in ["bottle", "fruit", "medicine"]:
            object_body = body
            object_label = cat
            print(f"  Found {cat} body: {body}")
            break

    drawer_joint_info = None
    if opened_joints:
        drawer_joint_info = opened_joints[0]
        print(f"  Drawer body: {drawer_joint_info['body']}")

    if object_body and drawer_joint_info:
        move_object_to_drawer(env, object_body, drawer_joint_info)
    else:
        print("  Warning: Could not find object or drawer body")

    # Re-observe after moving object
    # Need to move robot back so camera can see inside the drawer
    print("\n--- Re-observing after moving object ---")
    # Try multiple viewpoints to catch the bottle inside drawer
    view_candidates = [
        {"base": [3.0, 0.0, PI], "head": [0.0, 0.9], "name": "after_move_far"},
        {"base": [2.5, 0.5, PI], "head": [0.2, 0.9], "name": "after_move_side"},
    ]
    for vp in view_candidates:
        print(f"\n  Trying viewpoint: {vp['name']}")
        set_robot_viewpoint(env, vp)
        save_head_image(env, os.path.join(OUTPUT_DIR, f"view_{vp['name']}.png"))
        get_observations_update_memory(env, percept, memory, update_scene_graph=False)

    # Debug: check actual PyBullet position of moved object
    if object_body:
        actual_pos, actual_orn = p.getBasePositionAndOrientation(object_body)
        print(f"  Actual PyBullet position of bottle: {np.round(actual_pos, 3)}")

    # Verify real inside relation
    if object_label and drawer_joint_info:
        verify_real_inside_relation(memory, drawer_joint_info, object_label, object_body=object_body)

    # Also verify synthetic inside relation for comparison
    parent_node = memory.action_scene_graph.object_nodes.get("cabinet_gt_0")
    if parent_node is None:
        parent_node = memory.action_scene_graph.object_nodes.get("counter_gt_0")
    verify_inside_relation(memory, parent_node)

    # Additional low-angle viewpoints for oven/dishwasher detection
    print("\n--- Low-angle observations for oven/dishwasher ---")
    for i, vp in enumerate(LOW_VIEWPOINTS):
        print(f"\nLow viewpoint {i+1}/{len(LOW_VIEWPOINTS)}: {vp.get('name', i)}")
        set_robot_viewpoint(env, vp)
        save_head_image(env, os.path.join(OUTPUT_DIR, f"view_{vp.get('name', i)}.png"))
        get_observations_update_memory(env, percept, memory, update_scene_graph=False)

    # Save results
    print("\n--- Saving ---")
    sg_path = os.path.join(OUTPUT_DIR, "scene_graph.json")
    with open(sg_path, "w") as f:
        sg_data = {"nodes": {}}
        for nid, node in memory.action_scene_graph.object_nodes.items():
            sg_data["nodes"][nid] = {
                "label": node.node_label,
                "explored": getattr(node, "explored", False),
                "parent": node.parent.node_id if node.parent else None,
                "parent_relation": node.parent_relation,
            }
        json.dump(sg_data, f, indent=2)
    print(f"Scene graph: {sg_path}")

    memory_path = os.path.join(OUTPUT_DIR, "memory.pkl")
    memory.save_memory(memory_path)
    print(f"Memory: {memory_path}")

    if memory.action_scene_graph is not None:
        memory.action_scene_graph.visualize()
        print(f"Graphviz: {OUTPUT_DIR}/scene_graphs/")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
