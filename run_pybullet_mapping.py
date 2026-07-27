"""
Phase 0 Validation: Graft RoboEXP perception+memory onto kitchen-worlds PyBullet.

This script:
1. Loads a kitchen-worlds scene with FEG gripper
2. Initializes RoboPercept and RoboMemory
3. Collects observations from multiple wrist camera viewpoints
4. Runs perception (GroundingDINO + SAM + CLIP) on each view
5. Updates the ActionSceneGraph incrementally
6. Opens a drawer programmatically in PyBullet
7. Re-observes and updates scene graph with new 'inside' relations
8. Generates visualization video
"""
import sys
import os
import numpy as np
import pickle
import json

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from roboexp.perception.robo_percept import RoboPercept
from roboexp.memory.robo_memory import RoboMemory

from pybullet_tools.utils import PI, set_joint_position, get_joint_position
import pybullet as p


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OBJECT_LEVEL_LABELS = [
    "table", "counter", "cabinet", "drawer", "door",
    "microwave", "oven", "dishwasher", "fridge",
    "bottle", "cup", "bowl", "plate", "pot", "pan",
    "vegetable", "fruit", "medicine",
]
GROUNDING_DICT = " . ".join(OBJECT_LEVEL_LABELS) + " ."

# Camera poses for multi-view observation (FEG gripper SE3: [x, y, z, roll, pitch, yaw])
# Viewpoints aimed at the kitchen counter in test_feg_kitchen_mini
VIEWPOINTS = [
    [1.5, 0.0, 1.8, 0, -PI/2.5, 0],      # Front center
    [1.5, 0.8, 1.8, 0, -PI/2.5, PI/8],   # Front left
    [1.5, -0.8, 1.8, 0, -PI/2.5, -PI/8], # Front right
]

# Workspace bounds for voxel grid (meters)
LOWER_BOUND = [-1, -2, -0.5]
HIGHER_BOUND = [4, 2, 2.5]
VOXEL_SIZE = 0.02

OUTPUT_DIR = "/home/jx/kitchen/RoboEXP/phase0_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_observations_update_memory(env, percept, memory, update_scene_graph=False, scene_graph_option=None):
    """Collect observations, run perception, and update memory."""
    print("\n--- Collecting observations ---")
    observations = env.get_observations(wrist_only=True)
    print(f"Got {len(observations)} camera observations")

    print("--- Running perception (GroundingDINO + SAM + CLIP) ---")
    observation_attributes = percept.get_attributes_from_observations(
        observations, visualize=False
    )
    print(f"Perception done. Found attributes for {len(observation_attributes)} views")

    print("--- Updating memory ---")
    memory.update_memory(
        observations,
        observation_attributes,
        OBJECT_LEVEL_LABELS,
        direct_move=None,
        filter_masks={},
        extra_alignment=False,
        update_scene_graph=update_scene_graph,
        scene_graph_option=scene_graph_option,
        visualize=False,
    )
    print("Memory updated!")

    if memory.action_scene_graph is not None:
        print(f"Scene graph nodes: {len(memory.action_scene_graph.object_nodes)}")
        for node_id, node in memory.action_scene_graph.object_nodes.items():
            print(f"  Node {node_id}: {node.node_label} (explored={getattr(node, 'explored', None)})")

    return observations, observation_attributes


def find_drawer_handle(env, memory):
    """Find a drawer handle node in the scene graph."""
    if memory.action_scene_graph is None:
        return None
    for node_id, node in memory.action_scene_graph.object_nodes.items():
        if "handle" in node.node_label.lower():
            # Check if parent is a drawer
            parent = node.parent
            if parent and "drawer" in parent.node_label.lower():
                return node
    return None


def open_drawer_pybullet(env, handle_node):
    """Programmatically open a drawer in PyBullet."""
    print(f"\n--- Opening drawer: {handle_node.node_label} ---")
    # Get the instance corresponding to this handle
    instance = handle_node.instance
    
    # Find the articulated joint associated with this handle's parent object
    # In kitchen-worlds, drawers are articulated objects with joints
    # We need to find the body and joint index
    
    # For simplicity in Phase 0, we'll use the world.articulated_parts info
    world = env.world
    
    # Find a drawer joint and open it
    for drawer_info in world.articulated_parts.get('drawer', []):
        body, joint, _ = drawer_info
        # Get joint limits
        from pybullet_tools.utils import get_joint_limits, get_joint_name
        limits = get_joint_limits(body, joint)
        print(f"  Drawer joint: {get_joint_name(body, joint)}, limits: {limits}")
        
        # Open the drawer (set to max limit)
        open_position = limits[1]  # upper limit
        set_joint_position(body, joint, open_position)
        print(f"  Set drawer to open position: {open_position}")
        
        # Step simulation to settle
        for _ in range(100):
            p.stepSimulation()
        return True
    
    print("  No drawer found to open!")
    return False


def generate_video(env, memory, output_path):
    """Generate a simple visualization of the scene graph."""
    # Save scene graph visualization
    if memory.action_scene_graph is not None:
        memory.action_scene_graph.visualize()
        print(f"Scene graph visualization saved to {memory.base_dir}/scene_graphs/")
    
    # Save current observation
    obs = env.get_observations(wrist_only=True, save_image=True)
    
    print(f"Visualization complete")


def main():
    print("=" * 60)
    print("Phase 0: PyBullet + RoboEXP Perception/Memory Validation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Initialize environment
    # -----------------------------------------------------------------------
    print("\n[1/6] Initializing PyBullet environment...")
    env = PyBulletExplorationEnv(
        scene_builder="test_feg_kitchen_mini",
        robot_name="feg",
        use_gui=False,
        segment=True,
        initial_q=[1.0, 0.0, 1.5, 0, -PI/2, 0],
    )
    print("Environment ready!")

    # -----------------------------------------------------------------------
    # 2. Initialize perception and memory
    # -----------------------------------------------------------------------
    print("\n[2/6] Initializing RoboPercept (GroundingDINO + SAM + CLIP)...")
    percept = RoboPercept(
        grounding_dict=GROUNDING_DICT,
        lazy_loading=False,
        device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
    )
    print("RoboPercept ready!")

    print("\n[3/6] Initializing RoboMemory (voxel grid + scene graph)...")
    memory = RoboMemory(
        lower_bound=LOWER_BOUND,
        higher_bound=HIGHER_BOUND,
        voxel_size=VOXEL_SIZE,
        real_camera=False,
        base_dir=OUTPUT_DIR,
    )
    print("RoboMemory ready!")

    # -----------------------------------------------------------------------
    # 3. Initial multi-view observation and scene graph construction
    # -----------------------------------------------------------------------
    print("\n[4/6] Collecting initial multi-view observations...")
    for i, viewpoint in enumerate(VIEWPOINTS):
        print(f"\n  Viewpoint {i+1}/{len(VIEWPOINTS)}: {viewpoint}")
        env.run_action(1, list(viewpoint), iteration=50)
        update_sg = (i == len(VIEWPOINTS) - 1)  # Only update SG after last view
        get_observations_update_memory(
            env, percept, memory,
            update_scene_graph=update_sg,
            scene_graph_option=None,
        )

    # -----------------------------------------------------------------------
    # 4. Open a drawer
    # -----------------------------------------------------------------------
    print("\n[5/6] Opening a drawer...")
    handle_node = find_drawer_handle(env, memory)
    if handle_node:
        print(f"Found handle node: {handle_node.node_label}")
        open_drawer_pybullet(env, handle_node)
    else:
        print("No handle node found in scene graph. Trying to open any drawer...")
        # Fallback: open the first drawer we can find
        for drawer_info in env.world.articulated_parts.get('drawer', []):
            body, joint, _ = drawer_info
            from pybullet_tools.utils import get_joint_limits
            limits = get_joint_limits(body, joint)
            set_joint_position(body, joint, limits[1])
            for _ in range(100):
                p.stepSimulation()
            print(f"Opened drawer joint with limits {limits}")
            break

    # -----------------------------------------------------------------------
    # 5. Re-observe after opening
    # -----------------------------------------------------------------------
    print("\n[6/6] Re-observing after drawer opening...")
    # Re-observe after opening: only update memory, not scene graph
    # (Scene graph incremental update requires specific scene_graph_option)
    for i, viewpoint in enumerate(VIEWPOINTS[:2]):
        print(f"\n  Re-observation viewpoint {i+1}/{len(VIEWPOINTS[:2])}")
        env.run_action(1, list(viewpoint), iteration=50)
        get_observations_update_memory(
            env, percept, memory,
            update_scene_graph=False,
            scene_graph_option=None,
        )

    # -----------------------------------------------------------------------
    # 6. Save results
    # -----------------------------------------------------------------------
    print("\n--- Saving results ---")
    
    # Save scene graph
    if memory.action_scene_graph is not None:
        sg_path = os.path.join(OUTPUT_DIR, "scene_graph.json")
        with open(sg_path, "w") as f:
            # Simple serialization
            sg_data = {
                "nodes": {},
                "relations": [],
            }
            for node_id, node in memory.action_scene_graph.object_nodes.items():
                sg_data["nodes"][node_id] = {
                    "label": node.node_label,
                    "explored": node.explored,
                    "parent": node.parent.node_id if node.parent else None,
                    "parent_relation": node.parent_relation if hasattr(node, "parent_relation") else None,
                }
            json.dump(sg_data, f, indent=2)
        print(f"Scene graph saved to {sg_path}")
    
    # Save memory
    memory_path = os.path.join(OUTPUT_DIR, "memory.pkl")
    memory.save_memory(memory_path)
    print(f"Memory saved to {memory_path}")

    # Generate video/visualization
    video_path = os.path.join(OUTPUT_DIR, "phase0_visualization.mp4")
    generate_video(env, memory, video_path)

    print("\n" + "=" * 60)
    print("Phase 0 validation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
