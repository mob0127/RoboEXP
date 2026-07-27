"""Generate a Phase 0 validation video showing perception results on PyBullet images."""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.perception.robo_percept import RoboPercept
from roboexp.memory.robo_memory import RoboMemory

from pybullet_tools.utils import PI


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

VIEWPOINTS = [
    [1.5, 0.0, 1.8, 0, -PI/2.5, 0],
    [1.5, 0.8, 1.8, 0, -PI/2.5, PI/8],
    [1.5, -0.8, 1.8, 0, -PI/2.5, -PI/8],
]

OUTPUT_VIDEO = "/home/jx/kitchen/RoboEXP/phase0_outputs/phase0_demo.mp4"
FPS = 1


def draw_detections(img, obs_attrs):
    """Draw bounding boxes and labels on image."""
    img = (img * 255).clip(0, 255).astype(np.uint8).copy()
    if obs_attrs is None:
        return img
    
    boxes = obs_attrs.get("pred_boxes")
    phrases = obs_attrs.get("pred_phrases")
    masks = obs_attrs.get("pred_masks")
    
    if boxes is None or len(boxes) == 0:
        return img
    
    H, W = img.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        phrase = phrases[i] if phrases and i < len(phrases) else "?"
        label = str(phrase).split(":")[0] if ":" in str(phrase) else str(phrase)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Overlay mask if available
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask is not None and mask.shape == (H, W):
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0.5] = color
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.3, 0)
    
    return img


def draw_scene_graph_info(img, memory):
    """Overlay scene graph node info on image."""
    if memory.action_scene_graph is None:
        return img
    
    img = img.copy()
    y_offset = 30
    cv2.putText(img, "Scene Graph:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    for node_id, node in memory.action_scene_graph.object_nodes.items():
        text = f"  {node_id}: {node.node_label}"
        if hasattr(node, 'explored') and node.explored:
            text += " [E]"
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    return img


def main():
    print("Loading environment...")
    env = PyBulletExplorationEnv(
        scene_builder="test_feg_kitchen_mini",
        robot_name="feg",
        use_gui=False,
        segment=True,
        initial_q=[1.5, 0.0, 1.8, 0, -PI/2.5, 0],
    )
    
    print("Loading perception...")
    percept = RoboPercept(
        grounding_dict=GROUNDING_DICT,
        lazy_loading=False,
        device="cuda",
    )
    
    print("Loading memory...")
    memory = RoboMemory(
        lower_bound=LOWER_BOUND,
        higher_bound=HIGHER_BOUND,
        voxel_size=VOXEL_SIZE,
        real_camera=False,
    )
    
    frames = []
    
    for i, viewpoint in enumerate(VIEWPOINTS):
        print(f"\nViewpoint {i+1}/{len(VIEWPOINTS)}")
        env.run_action(1, list(viewpoint), iteration=50)
        
        # Get observations
        observations = env.get_observations(wrist_only=True)
        obs = observations["wrist"]
        rgb = obs["rgb"]  # (H, W, 3) float
        
        # Run perception
        obs_attrs = percept.get_attributes_from_observations(observations)
        attr = obs_attrs.get("wrist")
        
        # Update memory
        memory.update_memory(
            observations, obs_attrs, OBJECT_LEVEL_LABELS,
            update_scene_graph=(i == len(VIEWPOINTS) - 1),
            scene_graph_option=None,
            filter_masks={},
        )
        
        # Draw detections
        img = draw_detections(rgb, attr)
        
        # Draw scene graph info
        img = draw_scene_graph_info(img, memory)
        
        # Add viewpoint text
        cv2.putText(img, f"View: {viewpoint[:3]}", (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames.append(img)
    
    # Write video
    if frames:
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"\nVideo saved to: {OUTPUT_VIDEO}")
    
    env.close()


if __name__ == "__main__":
    main()
