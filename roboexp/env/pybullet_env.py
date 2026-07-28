# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
PyBulletExplorationEnv: A PyBullet-based environment adapter for RoboEXP.
Supports both FEG gripper and PR2 robot.
"""
import os
import sys
import numpy as np
import pybullet as p

KITCHEN_WORLDS = "/home/jx/kitchen/kitchen-worlds"
# Fallback copies that are reliably populated (the git submodules in
# kitchen-worlds are not always available).
KITCHEN_ROBOEXP = "/home/jx/kitchen/kitchen-roboexp"

# Insert kitchen-worlds paths first, then the fallback copies at the front so
# they take precedence when the kitchen-worlds submodules are empty/incomplete.
sys.path.insert(0, KITCHEN_WORLDS)
sys.path.insert(0, os.path.join(KITCHEN_WORLDS, "pybullet_planning"))
sys.path.insert(0, os.path.join(KITCHEN_WORLDS, "lisdf"))
sys.path.insert(0, os.path.join(KITCHEN_WORLDS, "pddlstream"))
sys.path.insert(0, os.path.join(KITCHEN_ROBOEXP, "pybullet_planning"))
sys.path.insert(0, os.path.join(KITCHEN_ROBOEXP, "lisdf"))
sys.path.insert(0, os.path.join(KITCHEN_ROBOEXP, "pddlstream"))

from pybullet_tools.utils import (
    connect, disconnect, set_camera_pose, unit_pose, point_from_pose,
    tform_point, get_pose, set_pose, Pose, Euler, PI, get_camera_matrix,
    multiply, invert, quat_from_euler, euler_from_quat, get_link_pose,
    link_from_name, set_joint_positions, wait_for_duration, HideOutput,
    LockRenderer, set_all_static, get_bodies,
    get_joint_info, get_joint_type, get_joint_name, get_joint_limits,
    get_aabb, add_fixed_constraint, remove_fixed_constraint, get_mass
)
from pybullet_tools.flying_gripper_utils import (
    set_gripper_positions, open_gripper, get_hand_pose, TOOL_LINK,
    get_joints_by_group, FINGERS_GROUP
)
from pybullet_tools.bullet_utils import (
    open_joint, close_joint, set_camera_target_body, CAMERA_MATRIX
)
from pybullet_tools.pr2_utils import (
    PR2_GROUPS, get_arm_joints, get_group_joints, get_carry_conf, set_arm_conf,
    set_group_conf, get_group_conf
)
from pybullet_tools.pr2_problems import create_pr2

from world_builder.world import World
from world_builder.entities import Camera, Object, Floor, Supporter
from world_builder.builders import (
    initialize_pybullet, get_world_builder, get_robot_builder
)
from world_builder.robot_builders import build_robot_from_args, create_pr2_robot


class PyBulletExplorationEnv:
    def __init__(
        self,
        scene_builder="test_feg_kitchen_mini",
        robot_name="pr2",  # "feg" or "pr2"
        camera_width=512,
        camera_height=512,
        camera_fx=256.0,
        camera_fy=256.0,
        max_depth=5.0,
        initial_base_q=(1.0, 0.0, PI),
        use_gui=False,
        segment=False,
        physical_grasp=False,  # FEG only: actuate fingers and create fixed constraint on close
    ):
        self.robot_name = robot_name
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_depth = max_depth
        self.use_gui = use_gui
        self.segment = segment
        self.initial_base_q = initial_base_q

        # Camera intrinsic
        self.camera_matrix = get_camera_matrix(
            width=camera_width, height=camera_height,
            fx=camera_fx, fy=camera_fy
        )
        cx, cy = (camera_width - 1) / 2.0, (camera_height - 1) / 2.0
        self.intrinsic = np.array([
            [camera_fx, 0, cx],
            [0, camera_fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Init PyBullet
        connect(use_gui=use_gui, shadows=True, width=1280, height=720)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(lightPosition=[5, 5, 10])

        # Create world
        self.world = World(time_step=1e-3, segment=segment)

        # Create robot FIRST
        if robot_name == "pr2":
            self.robot = create_pr2_robot(
                self.world, base_q=initial_base_q,
                custom_limits=((-2, -2, 0), (6, 6, 3))
            )
            # Set head to look slightly downward
            set_group_conf(self.robot.body, 'head', [0.0, 0.3])
            self.main_camera = self.robot.cameras[0]  # head camera
            self.camera_name = "head"
        else:
            initial_q = list(initial_base_q) + [0, -PI/2, 0]
            self.robot = build_robot_from_args(
                self.world, robot_name,
                initial_q=initial_q,
                custom_limits=((-2, -2, 0), (6, 6, 3))
            )
            self._add_wrist_camera()
            self.main_camera = self.wrist_camera
            self.camera_name = "wrist"

        # Load scene
        builder_fn = get_world_builder(scene_builder)
        builder_fn(self.world, verbose=False)

        set_all_static()

        # State
        self.gripper_state = 1
        self.current_observations = None
        self.target_pose = None

        # Physical grasp state (FEG only)
        self.physical_grasp = physical_grasp and (robot_name == "feg")
        if physical_grasp and robot_name != "feg":
            print(f"[PyBulletExplorationEnv] physical_grasp=True is only supported for robot_name='feg'; "
                  f"disabling physical grasp for '{robot_name}'.")
        self.tool_link = None
        self.grasped_body = None
        self.grasp_constraint = None
        self.grasp_radius = 0.20
        if self.physical_grasp:
            self.tool_link = link_from_name(self.robot.body, TOOL_LINK)
            open_gripper(self.robot.body)

    def _add_wrist_camera(self):
        camera_frame = "panda_hand"
        try:
            link_from_name(self.robot.body, camera_frame)
        except Exception:
            camera_frame = "base_link_0"
        camera = Camera(
            body=self.robot.body,
            camera_frame=camera_frame,
            camera_matrix=self.camera_matrix,
            max_depth=self.max_depth,
            name="wrist",
            draw_frame=camera_frame,
        )
        self.robot.cameras.append(camera)
        self.wrist_camera = camera

    def get_observations(self, wrist_only=True, save_image=False, **kwargs):
        """Capture from main camera (head for PR2, wrist for FEG)."""
        observations = {}
        cam = self.main_camera
        cam_img = cam.get_image(segment=self.segment, segment_links=False)
        rgb_rgba = cam_img.rgbPixels
        depth = cam_img.depthPixels
        seg = cam_img.segmentationMaskBuffer

        rgb = rgb_rgba[:, :, :3].astype(np.float32) / 255.0
        H, W = depth.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        
        # Get intrinsic from camera_matrix
        cam_matrix = cam.camera_matrix
        fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
        cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
        intrinsic = cam_matrix.astype(np.float32)
        
        z_eye = -depth
        x_eye = (uu - cx) * (-z_eye) / fx
        y_eye = (vv - cy) * (-z_eye) / fy
        position = np.stack([x_eye, y_eye, z_eye], axis=-1)
        mask = (depth > 0.01) & (depth < self.max_depth * 0.99)
        cam_pose = cam.get_pose()
        c2w = self._pose_to_matrix(cam_pose)

        observations[self.camera_name] = {
            "rgb": rgb,
            "position": position,
            "mask": mask,
            "c2w": c2w,
            "intrinsic": intrinsic,
        }
        if save_image:
            from PIL import Image
            img_uint8 = (rgb * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(f"{self.camera_name}_rgb.png")
        self.current_observations = observations
        return observations

    def _pose_to_matrix(self, pose):
        from pybullet_tools.utils import tform_from_pose
        return tform_from_pose(pose).astype(np.float32)

    def _get_tool_center(self):
        """Return the world position of the FEG tool link."""
        tool_pose = get_hand_pose(self.robot.body)
        return np.array(point_from_pose(tool_pose))

    def _detach_grasped_object(self):
        """Remove the active fixed constraint, if any."""
        if self.grasped_body is not None and self.tool_link is not None:
            remove_fixed_constraint(self.grasped_body, self.robot.body, self.tool_link)
        self.grasped_body = None
        self.grasp_constraint = None

    def _attach_nearby_object(self):
        """Create a fixed constraint between the FEG tool and a nearby object body."""
        if not self.physical_grasp or self.tool_link is None:
            return
        tool_center = self._get_tool_center()
        best_body, best_dist = None, self.grasp_radius
        for body in list(self.world.BODY_TO_OBJECT.keys()):
            if not isinstance(body, int):
                continue
            if body == self.robot.body:
                continue
            if get_mass(body) < 1e-6:
                continue
            aabb = get_aabb(body)
            center = (np.array(aabb[0]) + np.array(aabb[1])) / 2.0
            dist = np.linalg.norm(center - tool_center)
            if dist < best_dist:
                best_dist = dist
                best_body = body
        if best_body is not None:
            self.grasp_constraint = add_fixed_constraint(
                best_body, self.robot.body, robot_link=self.tool_link, max_force=None
            )
            self.grasped_body = best_body
            print(f"[physical_grasp] attached body {best_body} (dist={best_dist:.3f})")

    def run_action(self, action_code=0, action_parameters=[], iteration=100, **kwargs):
        """
        action_code:
            0: idle
            1: move base (PR2: [x, y, theta]) or move SE3 (FEG: [x,y,z,roll,pitch,yaw])
            2: open gripper
            3: close gripper
            4: reset
        """
        print(f"Action {action_code} params={action_parameters} iter={iteration}")
        if action_code == 0:
            self._step_simulation(iteration)
        elif action_code == 1:
            self.robot_move_to_pose(action_parameters)
            self._step_simulation(iteration)
        elif action_code == 2:
            self.gripper_state = 1
            if self.physical_grasp:
                self._detach_grasped_object()
                open_gripper(self.robot.body)
            self._step_simulation(iteration)
        elif action_code == 3:
            self.gripper_state = 0
            if self.physical_grasp:
                set_gripper_positions(self.robot.body, w=0.0)
                self._attach_nearby_object()
                self._step_simulation(iteration)
            else:
                self._step_simulation(iteration)
        elif action_code == 4:
            if self.physical_grasp:
                self._detach_grasped_object()
                open_gripper(self.robot.body)
            self.robot_reset()
            self._step_simulation(iteration)
        return True

    def robot_move_to_pose(self, pose):
        """
        PR2: pose = [x, y, theta] for base
        FEG: pose = [x, y, z, roll, pitch, yaw] for SE3
        """
        if self.robot_name == "pr2":
            x, y, theta = pose[:3]
            # PR2 uses base joints (x, y, theta) for planar movement.
            # resetBasePositionAndOrientation would break the internal coordinate system.
            from pybullet_tools.pr2_utils import set_group_conf
            set_group_conf(self.robot.body, 'base', [x, y, theta])
            self.target_pose = np.array(pose)
        else:
            from pybullet_tools.flying_gripper_utils import set_se3_conf
            set_se3_conf(self.robot.body, pose)
            self.target_pose = np.array(pose)
        return {"position": [pose]}

    def get_end_effector_pose(self):
        if self.robot_name == "pr2":
            pos, orn = p.getBasePositionAndOrientation(self.robot.body)
            euler = p.getEulerFromQuaternion(orn)
            return [pos[0], pos[1], euler[2]]
        else:
            from pybullet_tools.flying_gripper_utils import get_se3_conf
            return get_se3_conf(self.robot.body)

    def robot_reset(self):
        self.gripper_state = 1
        if self.robot_name == "pr2":
            x, y, theta = self.initial_base_q
            z = 0.0
            orn = p.getQuaternionFromEuler([0, 0, theta])
            p.resetBasePositionAndOrientation(self.robot.body, [x, y, z], orn)
        else:
            from pybullet_tools.flying_gripper_utils import set_se3_conf
            initial_q = list(self.initial_base_q) + [0, -PI/2, 0]
            set_se3_conf(self.robot.body, initial_q)
        self.target_pose = None

    def _step_simulation(self, steps):
        for _ in range(steps):
            p.stepSimulation()

    def get_global_image(self, width=640, height=480):
        """Capture a fixed bird's-eye view of the whole scene.

        This is useful for demos / global visualisation.  It does not move the
        robot wrist camera; it uses an independent PyBullet debug camera.
        """
        target = [0.5, 0.0, 0.5]
        view = p.computeViewMatrixFromYawPitchRoll(
            distance=3.5, yaw=45.0, pitch=-45.0, roll=0.0,
            cameraTargetPosition=target,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=float(width) / height, nearVal=0.1, farVal=10.0
        )
        img = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj)
        rgb = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb

    def get_articulated_joints(self):
        """Return list of dicts for drawers/doors from world objects."""
        result = []
        from world_builder.entities import Door, Drawer
        for obj in self.world.OBJECTS_BY_CATEGORY.get('door', []):
            if isinstance(obj, Door):
                body, joint = obj.body, obj.joint
                limits = get_joint_limits(body, joint)
                result.append({
                    'body': body, 'joint': joint, 'part_type': 'door',
                    'joint_type': 'revolute', 'joint_name': get_joint_name(body, joint),
                    'limits': limits, 'handle_link': obj.handle_link,
                })
        for obj in self.world.OBJECTS_BY_CATEGORY.get('drawer', []):
            if isinstance(obj, Drawer):
                body, joint = obj.body, obj.joint
                limits = get_joint_limits(body, joint)
                result.append({
                    'body': body, 'joint': joint, 'part_type': 'drawer',
                    'joint_type': 'prismatic', 'joint_name': get_joint_name(body, joint),
                    'limits': limits, 'handle_link': obj.handle_link,
                })
        return result

    def set_joint_position(self, body, joint, position):
        from pybullet_tools.utils import set_joint_position
        set_joint_position(body, joint, position)

    def close(self):
        disconnect()
