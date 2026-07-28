# Xiao : 新增文件，用于本仓库对上游的扩展。
"""Quick test for PR2 robot in PyBulletExplorationEnv."""
import sys
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.utils import PI

def test_pr2():
    print("Initializing PR2 environment...")
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=True,
        initial_base_q=(1.0, 0.0, PI),
    )
    print("PR2 loaded!")
    print(f"Cameras: {[c.name for c in env.robot.cameras]}")
    
    print("Getting observations from head camera...")
    obs = env.get_observations(save_image=True)
    print(f"Obs shape: {obs['head']['rgb'].shape}")
    
    print("Moving PR2 base...")
    env.run_action(1, [1.5, 0.5, PI/2], iteration=50)
    
    print("Getting observations after move...")
    obs2 = env.get_observations(save_image=True)
    print("Done!")
    
    joints = env.get_articulated_joints()
    print(f"\nArticulated joints found: {len(joints)}")
    for j in joints:
        print(f"  {j['joint_name']}: type={j['joint_type']}, limits={j['limits']}")
    
    env.close()

if __name__ == "__main__":
    test_pr2()
