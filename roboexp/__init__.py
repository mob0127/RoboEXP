# Xiao : 新增/修改：以下 39 行为相对 Jianghanxiao/RoboEXP 的改动。
try:
    from .env import RobotExplorationReal, RoboCalibrate
except ImportError as e:
    RobotExplorationReal = None
    RoboCalibrate = None
    # print(f"Warning: Could not import env modules: {e}")

try:
    from .perception import RoboPercept
except ImportError as e:
    RoboPercept = None
    # print(f"Warning: Could not import perception modules: {e}")

try:
    from .memory import RoboMemory
except ImportError as e:
    RoboMemory = None
    # print(f"Warning: Could not import memory modules: {e}")

try:
    from .act import RoboAct, RoboActReal
except ImportError as e:
    RoboAct = None
    RoboActReal = None
    # print(f"Warning: Could not import act modules: {e}")

try:
    from .decision import RoboDecision
except ImportError as e:
    RoboDecision = None
    # print(f"Warning: Could not import decision modules: {e}")

try:
    from .utils import get_pose_from_front_up_end_effector, display_image, display_denseCLIP
except ImportError as e:
    get_pose_from_front_up_end_effector = None
    display_image = None
    display_denseCLIP = None
    # print(f"Warning: Could not import utils modules: {e}")
