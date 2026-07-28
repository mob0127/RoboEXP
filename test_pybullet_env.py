# Xiao : 新增文件，用于本仓库对上游的扩展。
"""Quick test for PyBulletExplorationEnv."""
import sys
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP/roboexp/env")

from pybullet_env import PyBulletExplorationEnv

def test_basic():
    print("Initializing environment...")
    env = PyBulletExplorationEnv(
        scene_builder="test_feg_kitchen_mini",
        robot_name="feg",
        use_gui=False,
        segment=True,
    )
    print("Environment loaded!")

    print("Getting observations...")
    obs = env.get_observations(wrist_only=True, save_image=True)
    print(f"Observations keys: {obs.keys()}")

    for name, data in obs.items():
        print(f"\nCamera: {name}")
        print(f"  rgb shape: {data['rgb'].shape}, dtype: {data['rgb'].dtype}")
        print(f"  position shape: {data['position'].shape}, dtype: {data['position'].dtype}")
        print(f"  mask shape: {data['mask'].shape}, dtype: {data['mask'].dtype}")
        print(f"  c2w shape: {data['c2w'].shape}, dtype: {data['c2w'].dtype}")
        print(f"  intrinsic:\n{data['intrinsic']}")
        print(f"  mask True count: {data['mask'].sum()}")
        print(f"  depth range: [{data['position'][:,:,2].min():.3f}, {data['position'][:,:,2].max():.3f}]")

    print("\nMoving robot...")
    from pybullet_tools.utils import PI
    env.run_action(1, [1.5, 0.5, 1.8, 0, -PI/2, 0], iteration=50)

    print("Getting observations after move...")
    obs2 = env.get_observations(wrist_only=True, save_image=True)
    print("Done!")

    env.close()

if __name__ == "__main__":
    test_basic()
