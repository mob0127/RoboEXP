# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Experiment: Can the robot observe a cup changing position and update the scene graph?

Reference: Phase 0 FEG validation (generate_phase0_video.py).

Because RoboMemory's scene graph can only be built once per memory, we run two
independent phases with the same random seed so the kitchen layout is identical:
  Phase A: red cylinder (category "cup") on the shelf -> observe from Phase 0 viewpoints -> build scene graph.
  Phase B: same cylinder moved to the counter -> observe from Phase 0 viewpoints -> build scene graph.

Only the central viewpoint is used to update memory / build the scene graph;
the two side viewpoints are rendered for the video but do not feed the graph.

The zero-shot detector (GroundingDINO+SAM) does not reliably detect the small
red cylinder in this view, so we generate the instance mask directly from the
rendered color image (the cylinder is the only bright-red object).  The
remaining pipeline -- voxel fusion, instance matching, and scene-graph building
-- is unchanged.

NOTE: The color-based fallback used here is a PyBullet/simulation-only patch.
It relies on known rendered colors and does not generalize to real images or
unknown objects.  For a real robot, replace it with a domain-appropriate
detector (e.g. SAM automatic masks + CLIP).

Expected behavior under the current RoboMemory matcher (voxel IoU + CLIP similarity):
  - After the cup moves, the new observation has near-zero voxel overlap with the
    old instance, so it is registered as a NEW instance/node rather than updating
    the original. This demonstrates the identity-preservation problem.
"""
import sys
import os
import random
import json
import numpy as np
import cv2

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
# Use the kitchen-roboexp copy of pybullet_planning because the submodule in
# kitchen-worlds is not reliably populated.
sys.path.insert(0, "/home/jx/kitchen/kitchen-roboexp/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.memory.robo_memory import RoboMemory
from roboexp.perception.robo_percept import RoboPercept
from roboexp.perception.pybullet_fallback import PyBulletFallbackDetector

from pybullet_tools.utils import PI, get_aabb, set_pose, Pose, Point, create_cylinder, RED
from world_builder.entities import Object
import pybullet as p

CYLINDER_RADIUS = 0.08
CYLINDER_HEIGHT = 0.20

OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
    "handle", "knob", "button",
]

# Simulation-only PyBullet color fallback config for objects that DINO+SAM misses.
# These RGB thresholds are tied to the rendered colors of this synthetic scene;
# do not use on real images.
PYBULLET_FALLBACK_CONFIG = {
    "cup": {
        "lower": [0.25, 0.00, 0.00],   # rendered red cylinder lower bound
        "upper": [1.00, 0.20, 0.15],   # rendered red cylinder upper bound
        "morph_kernel": 3,
    },
}
FALLBACK_LABELS = ["cup"]

LOWER_BOUND = [-1, -2, -0.5]
HIGHER_BOUND = [4, 2, 2.5]
VOXEL_SIZE = 0.02

# Same viewpoints as Phase 0 FEG validation
VIEWPOINTS = [
    [1.5, 0.0, 1.8, 0, -PI / 2.5, 0],
    [1.5, 0.8, 1.8, 0, -PI / 2.5, PI / 8],
    [1.5, -0.8, 1.8, 0, -PI / 2.5, -PI / 8],
]

OUT_DIR = "/home/jx/kitchen/RoboEXP/experiments/outputs"
TASK_DIR = os.path.join(OUT_DIR, "task3_cup_identity")
VIDEO_PATH = os.path.join(TASK_DIR, "cup_move_scene_graph.mp4")
REPORT_PATH = os.path.join(TASK_DIR, "cup_move_report.json")
FRAME_DIR = os.path.join(TASK_DIR, "frames")
SAVE_DEMO_FRAMES = True
FPS = 1

# Dimensionality of the CLIP features used by RoboMemory for matching.
MASK_FEAT_DIM = 512


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_demo_frame(rgb, tag, mask=None):
    """Save a raw RGB (and optional mask overlay) frame for the overall demo."""
    if not SAVE_DEMO_FRAMES:
        return
    ensure_dir(FRAME_DIR)
    img = (rgb * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(FRAME_DIR, f"{tag}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if mask is not None:
        overlay = img.copy()
        overlay[mask] = (255, 0, 0)
        cv2.imwrite(os.path.join(FRAME_DIR, f"{tag}_mask.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_global_demo_frame(env, tag, label=None):
    """Save a global bird's-eye view frame with an optional label."""
    if not SAVE_DEMO_FRAMES:
        return
    ensure_dir(FRAME_DIR)
    rgb = env.get_global_image()
    if label:
        rgb = cv2.putText(np.ascontiguousarray(rgb), label, (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FRAME_DIR, f"{tag}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def add_test_cup(env, surface_top_z, pos, name="test_cup"):
    """Create a bright red cylinder and register it as category 'cup'."""
    z = pos[2]
    if z is None:
        z = surface_top_z + CYLINDER_HEIGHT / 2 + 0.01
    body = create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, color=RED)
    cup = env.world.add_object(Object(body, category="cup", name=name))
    set_pose(cup.body, Pose(point=Point(pos[0], pos[1], z)))
    for _ in range(20):
        p.stepSimulation()
    print(f"Added cup '{cup.name}' body={cup.body} at [{pos[0]:.3f}, {pos[1]:.3f}, {z:.3f}]")
    return cup


def red_object_mask(rgb):
    """Segment the bright-red cylinder from the rendered RGB image."""
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 0.25) & (red > green + 0.15) & (red > blue + 0.15)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def make_cup_attributes(obs):
    """
    Construct observation_attributes for a single 'cup' detection from the
    red-cylinder color mask.  Uses a constant (non-zero) CLIP feature vector.
    """
    mask = red_object_mask(obs["rgb"])
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "pred_boxes": np.zeros((0, 4), dtype=np.float32),
            "pred_phrases": [],
            "pred_masks": np.zeros((0, *mask.shape), dtype=bool),
            "mask_feats": np.zeros((0, MASK_FEAT_DIM), dtype=np.float32),
        }
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    masks = mask[None, ...]  # (1, H, W)
    # A non-zero constant vector; identical vectors have cosine similarity 1,
    # but the matcher still requires voxel overlap, so disjoint objects will not
    # be spuriously merged.
    feats = np.ones((1, MASK_FEAT_DIM), dtype=np.float32) * 1e-3
    return {
        "pred_boxes": box,
        "pred_phrases": ["cup:1.0"],
        "pred_masks": masks,
        "mask_feats": feats,
    }


def draw_detections(img, box, phrase="cup:1.0"):
    """Draw a bounding box and label for the manually generated detection."""
    img = (img * 255).clip(0, 255).astype(np.uint8).copy()
    if box is None or len(box) == 0:
        return img
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, phrase, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)
    return img


def summarize_cup_nodes(memory, target_label="cup"):
    """Extract JSON-serializable info about all cup nodes in the scene graph."""
    sg = memory.action_scene_graph
    if sg is None:
        return []
    cups = []
    for node_id, node in sg.object_nodes.items():
        if node.node_label != target_label:
            continue
        voxels = getattr(node.instance, "voxel_indexes", None)
        if voxels is None or len(voxels) == 0:
            continue
        attrs = node.instance.get_attributes()
        cups.append({
            "node_id": node_id,
            "instance_id": str(node.instance.instance_id),
            "label": node.node_label,
            "center": [round(float(v), 4) for v in attrs["center"]],
            "size": [round(float(v), 4) for v in attrs["size"]],
            "parent": node.parent.node_id if node.parent else None,
            "parent_relation": node.parent_relation,
        })
    return cups


def draw_scene_graph_panel(img, memory, stage, target_label="cup"):
    """Overlay scene-graph object list, highlighting target label nodes."""
    img = img.copy()
    y = 25
    cv2.putText(img, f"Stage: {stage}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    y += 25
    cv2.putText(img, "Scene Graph:", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    y += 22

    sg = memory.action_scene_graph
    if sg is None:
        cv2.putText(img, "  (not built)", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        return img

    for node_id, node in sg.object_nodes.items():
        voxels = getattr(node.instance, "voxel_indexes", None)
        if node.instance is None or voxels is None or len(voxels) == 0:
            text = f"{node_id}: {node.node_label} (no voxels)"
            color = (128, 128, 128)
        else:
            attrs = node.instance.get_attributes()
            center = attrs["center"]
            is_target = node.node_label == target_label
            color = (0, 255, 0) if is_target else (200, 200, 200)
            text = f"{node_id}: {node.node_label} @ [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
        if y > img.shape[0] - 10:
            break
    return img


def get_surface_zs(seed=42):
    """Quickly load the scene and return the top z of shelf and counter."""
    random.seed(seed)
    np.random.seed(seed)
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="feg",
        use_gui=False,
        segment=False,
        initial_base_q=(1.5, 0.0, 1.8),
    )
    env.robot_move_to_pose([1.5, 0.0, 1.8, 0, -PI / 2.5, 0])
    env.run_action(0, [], iteration=50)
    shelf_z = get_aabb(env.world.name_to_object("shelf").body)[1][2]
    counter_z = get_aabb(env.world.name_to_object("counter").body)[1][2]
    env.close()
    return shelf_z, counter_z


def observe_phase(
    env, memory, percept, fallback_detector,
    stage_name, scene_graph_option=None
):
    """
    Move the wrist camera through VIEWPOINTS, feed only the central viewpoint
    into memory, and return rendered frames + cup node summary.
    """
    frames = []
    box_to_draw = None
    for i, vp in enumerate(VIEWPOINTS):
        print(f"\n[{stage_name}] viewpoint {vp[:3]}")
        env.run_action(1, list(vp), iteration=50)
        observations = env.get_observations(wrist_only=True)
        obs = observations["wrist"]

        # Run DINO+SAM, then replace the PyBullet synthetic-object labels with
        # the color-based fallback masks.  DINO often gives false positives for
        # these low-polygon objects, so we force the fallback for those labels.
        obs_attrs = percept.get_attributes_with_fallback(
            observations, fallback_detector, FALLBACK_LABELS,
            replace_existing=True
        )
        # Draw the first "cup" detection if available.
        cup_box = None
        for j, phrase in enumerate(obs_attrs["wrist"].get("pred_phrases", [])):
            if str(phrase).split(":")[0] == "cup":
                cup_box = obs_attrs["wrist"]["pred_boxes"][j]
                break
        box_to_draw = cup_box

        # Only the central viewpoint feeds memory / scene graph.
        update_sg = (i == 0)
        if update_sg:
            memory.update_memory(
                observations, obs_attrs, OBJECT_LEVEL_LABELS,
                update_scene_graph=True,
                scene_graph_option=scene_graph_option,
                filter_masks={},
            )
            # Save raw RGB and fallback mask overlay for the overall demo.
            cup_mask = None
            for j, phrase in enumerate(obs_attrs["wrist"].get("pred_phrases", [])):
                if str(phrase).split(":")[0] == "cup":
                    cup_mask = obs_attrs["wrist"]["pred_masks"][j]
                    break
            save_demo_frame(obs["rgb"], f"task3_{stage_name.lower()}_central", mask=cup_mask)
            save_global_demo_frame(env, f"task3_{stage_name.lower()}_global",
                                   label=f"{stage_name}: global view")

        img = draw_detections(obs["rgb"], box_to_draw)
        img = draw_scene_graph_panel(img, memory, stage_name)
        cv2.putText(img, f"view={vp[:3]}", (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        frames.append(img)

    cups = summarize_cup_nodes(memory)
    print(f"[{stage_name}] cup nodes:", json.dumps(cups, indent=2))
    return frames, cups


def main():
    ensure_dir(TASK_DIR)

    # Cup (red cylinder) moves from the shelf to the counter (large, clearly visible displacement).
    initial_cup_pos = [0.25, 0.00, None]
    moved_cup_pos = [0.60, 0.00, None]

    shelf_z, counter_z = get_surface_zs()
    initial_cup_pos[2] = shelf_z + CYLINDER_HEIGHT / 2 + 0.01
    moved_cup_pos[2] = counter_z + CYLINDER_HEIGHT / 2 + 0.01

    random.seed(42)
    np.random.seed(42)

    print("\n==================== SETUP ====================")
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="feg",
        use_gui=False,
        segment=False,
        initial_base_q=(1.5, 0.0, 1.8),
    )
    env.robot_move_to_pose([1.5, 0.0, 1.8, 0, -PI / 2.5, 0])
    env.run_action(0, [], iteration=50)

    # Single memory with dynamic position-based re-association enabled.
    memory = RoboMemory(
        lower_bound=LOWER_BOUND,
        higher_bound=HIGHER_BOUND,
        voxel_size=VOXEL_SIZE,
        real_camera=False,
        position_association_enabled=True,
        position_association_threshold=1.0,
        move_merge_distance_threshold=0.2,
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    percept = RoboPercept(grounding_dict="cup . table . counter .", lazy_loading=False, device=device)
    fallback_detector = PyBulletFallbackDetector(PYBULLET_FALLBACK_CONFIG)

    shelf = env.world.name_to_object("shelf")
    surface_top_z = get_aabb(shelf.body)[1][2]
    cup = add_test_cup(env, surface_top_z, initial_cup_pos)

    # Phase A: observe at the initial position and initialize the scene graph.
    frames_before, cups_before = observe_phase(
        env, memory, percept, fallback_detector, "BEFORE_MOVE", scene_graph_option=None
    )

    # Move the cup to the counter inside the same environment.
    print("\n==================== MOVING CUP ====================")
    set_pose(cup.body, Pose(point=Point(*moved_cup_pos)))
    for _ in range(20):
        p.stepSimulation()
    print(f"Cup moved to {moved_cup_pos}")

    # Phase B: observe again and reassociate existing scene-graph nodes.
    frames_after, cups_after = observe_phase(
        env, memory, percept, fallback_detector, "AFTER_MOVE",
        scene_graph_option={"type": "reassociate"},
    )

    env.close()

    report = {
        "initial_cup_position": initial_cup_pos,
        "moved_cup_position": moved_cup_pos,
        "viewpoints": VIEWPOINTS,
        "stages": [
            {"name": "before_move", "cup_nodes": cups_before},
            {"name": "after_move", "cup_nodes": cups_after},
        ],
        "num_cup_nodes_before": len(cups_before),
        "num_cup_nodes_after": len(cups_after),
        "identity_preserved": bool(
            cups_before and cups_after and
            {n["instance_id"] for n in cups_before} == {n["instance_id"] for n in cups_after}
        ),
    }

    all_frames = frames_before + frames_after
    if all_frames:
        H, W = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (W, H))
        for frame in all_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"\nVideo saved: {VIDEO_PATH}")

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {REPORT_PATH}")

    # Save the final scene graph in a phase0-compatible JSON format.
    sg_path = os.path.join(TASK_DIR, "scene_graph.json")
    if memory.action_scene_graph is not None:
        with open(sg_path, "w") as f:
            json.dump(memory.action_scene_graph.to_dict(), f, indent=2)
        print(f"Scene graph saved: {sg_path}")

    print("\n========== CONCLUSION ==========")
    print(json.dumps(report, indent=2))
    if report["identity_preserved"]:
        print("Cup identity WAS preserved: the moved cup remained the same scene-graph node/instance.")
    else:
        print("Cup identity was NOT preserved: the moved cup became a new scene-graph node/instance.")


if __name__ == "__main__":
    main()
