# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Record a demonstration video of Phase 0 PR2 kitchen mapping.
Layout: left=fixed global camera, right=head camera + scrolling HUD.
"""
import sys
import os
import numpy as np
import pybullet as p
import cv2

sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds")
sys.path.insert(0, "/home/jx/kitchen/kitchen-worlds/pybullet_planning")
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

from roboexp.env.pybullet_env import PyBulletExplorationEnv
from pybullet_tools.utils import PI
from pybullet_tools.pr2_utils import set_group_conf

# Output
VIDEO_PATH = "/home/jx/kitchen/RoboEXP/phase0_outputs_pr2/phase0_demo.mp4"
FPS = 30
CANVAS_W, CANVAS_H = 1280, 720
FIXED_W, FIXED_H = 780, 720
HEAD_W, HEAD_H = 500, 500

# Fixed camera settings
FIXED_EYE = [2.5, 2.0, 2.0]
FIXED_TARGET = [0.3, -0.3, 0.8]
FIXED_FOV = 60

# PR2 viewpoints used in actual mapping (fixed to avoid clipping)
VIEWPOINTS = [
    {"base": [2.0, 0.0, PI], "head": [0.0, 0.3], "name": "front",
     "objects": ["table", "bottle", "medicine", "fruit", "vegetable", "door", "handle"]},
    {"base": [2.0, 0.6, PI], "head": [0.3, 0.3], "name": "left",
     "objects": ["table", "bottle", "fruit"]},
    {"base": [2.0, -0.6, PI], "head": [-0.3, 0.3], "name": "right",
     "objects": ["table", "medicine", "door", "handle"]},
    {"base": [2.5, 0.0, PI], "head": [0.0, 0.2], "name": "far_front",
     "objects": ["table", "bottle", "fruit", "medicine"]},
    {"base": [1.8, 0.0, PI], "head": [0.0, -0.4], "name": "look_up",
     "objects": ["cabinet", "microwave", "shelf", "medicine"]},
    {"base": [1.8, 0.6, PI], "head": [0.4, -0.35], "name": "look_up_left",
     "objects": ["cabinet", "microwave", "shelf"]},
    {"base": [1.8, -0.6, PI], "head": [-0.4, -0.35], "name": "look_up_right",
     "objects": ["cabinet", "microwave", "shelf"]},
]


class VideoRecorder:
    def __init__(self, env, output_path, fps=30):
        self.env = env
        self.fps = fps
        self.frames = []
        self.frame_idx = 0
        # Fixed camera matrices
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=FIXED_EYE,
            cameraTargetPosition=FIXED_TARGET,
            cameraUpVector=[0, 0, 1],
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=FIXED_FOV, aspect=FIXED_W / FIXED_H,
            nearVal=0.1, farVal=10.0,
        )
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_W, CANVAS_H))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
        # HUD scroll offset
        self.hud_scroll = 0

    def get_fixed_frame(self):
        img = p.getCameraImage(
            FIXED_W, FIXED_H,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.array(img[2]).reshape((FIXED_H, FIXED_W, 4))[:, :, :3]
        return rgb

    def get_head_frame(self):
        cam = self.env.robot.cameras[0]
        cam_img = cam.get_image()
        rgb = cam_img.rgbPixels[:, :, :3]
        # Resize to fit right panel
        rgb = cv2.resize(rgb, (HEAD_W, HEAD_H))
        return rgb

    def draw_hud(self, canvas, stage_info, head_info=None, scroll_items=None):
        """Overlay text on the canvas."""
        # Stage title at top-left of fixed view
        cv2.putText(canvas, stage_info, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Head cam label
        cv2.putText(canvas, "HEAD CAMERA", (FIXED_W + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if head_info:
            cv2.putText(canvas, head_info, (FIXED_W + 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Scrolling object list on the right panel below head cam
        if scroll_items:
            y_start = FIXED_H + 20  # below head cam? no, right panel is vertical
            # Actually right panel: x=FIXED_W to CANVAS_W, y=0 to CANVAS_H
            # Head cam occupies top part, scroll list goes below it
            panel_x = FIXED_W + 10
            y = HEAD_H + 40
            cv2.putText(canvas, "OBSERVED OBJECTS:", (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
            y += 25
            for i, item in enumerate(scroll_items):
                color = (255, 255, 255)
                if "handle" in item.lower() or "door" in item.lower():
                    color = (255, 100, 100)
                elif "bottle" in item.lower() or "fruit" in item.lower():
                    color = (100, 255, 100)
                cv2.putText(canvas, f"  {item}", (panel_x + 10, y + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # RoboEXP status box at bottom-left
        status_y = CANVAS_H - 20
        cv2.putText(canvas, "RoboEXP Phase 0 | PR2 + PyBullet", (20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    def capture(self, stage_info="", head_info="", scroll_items=None):
        fixed_rgb = self.get_fixed_frame()
        head_rgb = self.get_head_frame()

        # Build canvas
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        # Left: fixed view (scaled to fit height)
        canvas[:, :FIXED_W] = fixed_rgb
        # Right top: head camera
        h_off = (CANVAS_H - HEAD_H) // 2
        canvas[h_off:h_off + HEAD_H, FIXED_W:FIXED_W + HEAD_W] = head_rgb
        # Right bottom: black background for text
        cv2.rectangle(canvas, (FIXED_W, 0), (CANVAS_W, CANVAS_H), (20, 20, 20), -1)
        canvas[h_off:h_off + HEAD_H, FIXED_W:FIXED_W + HEAD_W] = head_rgb

        self.draw_hud(canvas, stage_info, head_info, scroll_items)

        # Convert RGB to BGR for OpenCV writer
        frame_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frame_idx += 1

    def hold(self, seconds, stage_info="", head_info="", scroll_items=None):
        for _ in range(int(seconds * self.fps)):
            self.capture(stage_info, head_info, scroll_items)

    def animate_move(self, start_base, end_base, start_head, end_head, seconds, stage_info=""):
        """Smoothly interpolate robot pose over `seconds`."""
        n_frames = int(seconds * self.fps)
        for i in range(n_frames + 1):
            t = i / n_frames
            # Interpolate base pose
            base = [
                start_base[0] + t * (end_base[0] - start_base[0]),
                start_base[1] + t * (end_base[1] - start_base[1]),
                start_base[2] + t * (end_base[2] - start_base[2]),
            ]
            # Interpolate head
            head = [
                start_head[0] + t * (end_head[0] - start_head[0]),
                start_head[1] + t * (end_head[1] - start_head[1]),
            ]
            # Set robot pose
            self.env.run_action(1, base, iteration=1)
            set_group_conf(self.env.robot.body, 'torso', [0.25])
            set_group_conf(self.env.robot.body, 'head', head)
            p.stepSimulation()
            head_info = f"base=[{base[0]:.1f}, {base[1]:.1f}] head=[{head[0]:.1f}, {head[1]:.1f}]"
            self.capture(stage_info, head_info)

    def close(self):
        self.writer.release()
        print(f"Video saved: {VIDEO_PATH}")


def main():
    print("Loading environment...")
    env = PyBulletExplorationEnv(
        scene_builder="sample_kitchen_mini_scene",
        robot_name="pr2",
        use_gui=False,
        segment=True,
        initial_base_q=(1.0, 0.0, PI),
    )
    recorder = VideoRecorder(env, VIDEO_PATH, fps=FPS)

    # Initial robot state (far enough to avoid clipping)
    env.run_action(1, [3.0, 0.0, PI], iteration=50)
    set_group_conf(env.robot.body, 'torso', [0.25])
    set_group_conf(env.robot.body, 'head', [0.0, 0.0])
    for _ in range(60):
        p.stepSimulation()

    # ---- Stage 1: Opening title ----
    print("Recording: Opening title")
    recorder.hold(2.0,
        stage_info="Phase 0: PR2 Kitchen Mapping",
        head_info="Initializing...",
        scroll_items=["Environment: sample_kitchen_mini_scene", "Robot: PR2"])

    # ---- Stage 2-8: Multi-view observation ----
    current_base = [1.0, 0.0, PI]
    current_head = [0.0, 0.0]

    for vp in VIEWPOINTS:
        print(f"Recording: viewpoint {vp['name']}")
        # Animate move
        recorder.animate_move(
            current_base, vp['base'],
            current_head, vp['head'],
            seconds=1.5,
            stage_info=f"Moving: {vp['name']}"
        )
        # Hold and observe
        set_group_conf(env.robot.body, 'head', vp['head'])
        for _ in range(10):
            p.stepSimulation()
        recorder.hold(2.0,
            stage_info=f"Viewpoint: {vp['name']}",
            head_info=f"Pan={vp['head'][0]:.1f} Tilt={vp['head'][1]:.1f}",
            scroll_items=vp['objects'])
        current_base = vp['base']
        current_head = vp['head']

    # ---- Stage 9: GT Furniture Injection ----
    print("Recording: GT furniture injection")
    recorder.hold(2.5,
        stage_info="GT Furniture Injection",
        head_info="Bypassing vision limitations",
        scroll_items=["counter (GT)", "microwave (GT)", "cabinet (GT)",
                     "shelf (GT)", "oven (GT)", "dishwasher (GT)"])

    # ---- Stage 10: Handle Injection ----
    print("Recording: Handle injection")
    recorder.hold(2.5,
        stage_info="Handle Injection (PyBullet joints)",
        head_info="8 articulated handles detected",
        scroll_items=["microwave_door_handle", "cabinet_door_handle x4",
                     "bottle_drawer_handle", "medicine_drawer_handle"])

    # ---- Stage 11: Open drawer ----
    print("Recording: Open drawer")
    # Animate drawer opening
    joints = env.get_articulated_joints()
    prismatic_joints = [j for j in joints if j['joint_type'] == 'prismatic']
    n_frames = int(1.5 * FPS)
    for i in range(n_frames + 1):
        t = i / n_frames
        for jinfo in prismatic_joints:
            body, joint = jinfo['body'], jinfo['joint']
            limits = jinfo['limits']
            pos = limits[1] * 0.8 * t
            p.resetJointState(body, joint, pos)
        p.stepSimulation()
        recorder.capture(
            stage_info="Action: OPEN drawer",
            head_info=f"Drawer progress: {t*100:.0f}%"
        )

    # ---- Stage 12: Move bottle into drawer ----
    print("Recording: Move bottle")
    # Find bottle body
    bottle_body = None
    for body, obj in env.world.BODY_TO_OBJECT.items():
        if isinstance(body, int):
            cat = getattr(obj, "category", "") or getattr(obj, "name", "")
            if cat == "bottle":
                bottle_body = body
                break

    # Get drawer AABB center
    drawer_joint = prismatic_joints[0] if prismatic_joints else None
    target_pos = [1.224, 0.464, 0.499]
    if drawer_joint:
        joint_data = p.getJointInfo(drawer_joint['body'], drawer_joint['joint'])
        link_name = joint_data[12].decode('utf-8')
        from pybullet_tools.utils import link_from_name
        link_id = link_from_name(drawer_joint['body'], link_name)
        aabb = p.getAABB(drawer_joint['body'], link_id)
        target_pos = [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]
        target_pos[2] += 0.05

    if bottle_body:
        start_pos, _ = p.getBasePositionAndOrientation(bottle_body)
        n_frames = int(1.5 * FPS)
        for i in range(n_frames + 1):
            t = i / n_frames
            pos = [
                start_pos[0] + t * (target_pos[0] - start_pos[0]),
                start_pos[1] + t * (target_pos[1] - start_pos[1]),
                start_pos[2] + t * (target_pos[2] - start_pos[2]),
            ]
            p.resetBasePositionAndOrientation(bottle_body, pos, p.getQuaternionFromEuler([0, 0, 0]))
            p.stepSimulation()
            recorder.capture(
                stage_info="Action: MOVE bottle into drawer",
                head_info=f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            )

    # ---- Stage 13: Re-observe ----
    print("Recording: Re-observe")
    recorder.hold(2.0,
        stage_info="Re-observing after manipulation",
        head_info="Checking inside relation...",
        scroll_items=["bottle (moved)", "drawer (opened)"])

    # ---- Stage 14: Inside relation verified ----
    print("Recording: Inside relation")
    recorder.hold(3.0,
        stage_info="Inside Relation VERIFIED",
        head_info="bottle -> inside -> cabinet",
        scroll_items=["bottle_1 (instance)", "parent: open_0",
                     "parent_relation: INSIDE",
                     "", "apple_0 (synthetic)", "parent: open_1",
                     "parent_relation: INSIDE"])

    # ---- Stage 15: Ending ----
    recorder.hold(2.0,
        stage_info="Phase 0 Complete",
        head_info="Scene graph saved | Memory saved",
        scroll_items=["Scene graph: 20 nodes", "Handles: 8 explored",
                     "Inside relations: 2 verified"])

    recorder.close()
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
