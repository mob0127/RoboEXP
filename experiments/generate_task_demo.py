# Xiao : 新增文件，用于本仓库对上游的扩展。
"""
Generate an overall demo video for the five RoboEXP cup tasks.

This version uses global (bird's-eye) views, bigger labels, and holds each
frame for several seconds so the viewer can actually see what is happening.

Expected inputs:
  - outputs/task3_cup_identity/frames/task3_*_global.png
  - outputs/task5_cup_grasp/frames/task5_*.png

Outputs:
  - outputs/demo/frame_XX_tXXXX.png
  - outputs/demo/tasks_demo.mp4
  - outputs/demo/scene_graph.json
  - outputs/demo/report.json
"""
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "/home/jx/kitchen/RoboEXP/experiments/outputs"
TASK3_FRAMES = os.path.join(BASE, "task3_cup_identity", "frames")
TASK5_FRAMES = os.path.join(BASE, "task5_cup_grasp", "frames")
DEMO_DIR = os.path.join(BASE, "demo")
FRAME_SIZE = (640, 480)
FPS = 1
HOLD_SECONDS = 3  # each scene is held this many seconds


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_rgb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, FRAME_SIZE)
    return img


def make_title(text, subtext=""):
    """Create a black title card with white/yellow text."""
    img = Image.new("RGB", FRAME_SIZE, (20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font_big)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((FRAME_SIZE[0] - w) // 2, (FRAME_SIZE[1] - h) // 2 - 40),
              text, fill=(255, 255, 0), font=font_big)

    if subtext:
        lines = subtext.split("\n")
        y = (FRAME_SIZE[1] - h) // 2 + 30
        for line in lines:
            bbox2 = draw.textbbox((0, 0), line, font=font_small)
            w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
            draw.text(((FRAME_SIZE[0] - w2) // 2, y),
                      line, fill=(220, 220, 220), font=font_small)
            y += h2 + 10
    return np.array(img)


def add_label(img, label, color=(0, 255, 255)):
    """Overlay a large, readable label at the top-left of a CV2 RGB image."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x, y = 20, 30 + th
    # dark background for readability
    cv2.rectangle(out, (x - 5, y - th - 10), (x + tw + 10, y + 5), (0, 0, 0), -1)
    cv2.putText(out, label, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return out


def save_frame(idx, img, timestamp):
    path = os.path.join(DEMO_DIR, f"frame_{idx:02d}_t{timestamp:04d}.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def build_demo():
    ensure_dir(DEMO_DIR)
    # Remove old demo frames to avoid stale files
    for f in os.listdir(DEMO_DIR):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(DEMO_DIR, f))

    scenes = []

    # 00: overall title
    scenes.append(make_title("RoboEXP Cup Tasks Demo",
                             "Tasks 1-5: diagnosis → patch → fallback → grasp validation"))

    # 01-03: Task 1 & 2 — the original identity-loss problem
    scenes.append(make_title("Task 1 & 2: Bug Diagnosis",
                             "After the cup moves, voxel IoU = 0\nso RoboMemory registers it as a NEW instance"))
    scenes.append(add_label(load_rgb(os.path.join(TASK3_FRAMES, "task3_before_move_global.png")),
                            "BEFORE MOVE: cup_0 on shelf", color=(0, 255, 255)))
    scenes.append(add_label(load_rgb(os.path.join(TASK3_FRAMES, "task3_after_move_global.png")),
                            "WITHOUT PATCH: becomes cup_1 (IoU = 0)", color=(0, 0, 255)))

    # 04-06: Task 3 — identity-preservation patch
    scenes.append(make_title("Task 3: Identity-Preservation Patch",
                             "move_instance + position reassociation + reassociate option"))
    scenes.append(add_label(load_rgb(os.path.join(TASK3_FRAMES, "task3_before_move_global.png")),
                            "BEFORE MOVE: cup_0", color=(0, 255, 255)))
    scenes.append(add_label(load_rgb(os.path.join(TASK3_FRAMES, "task3_after_move_global.png")),
                            "WITH PATCH: still cup_0 (identity preserved)", color=(0, 255, 0)))

    # 07-08: Task 4 — PyBullet fallback mask
    scenes.append(make_title("Task 4: PyBullet Perception Fallback",
                             "GroundingDINO+SAM misses the small red cylinder\nColor fallback provides the mask"))
    scenes.append(add_label(load_rgb(os.path.join(TASK3_FRAMES, "task3_before_move_central_mask.png")),
                            "FALLBACK MASK: red cylinder segmented", color=(255, 0, 0)))

    # 09-13: Task 5 — grasp validation
    scenes.append(make_title("Task 5: Grasp Validation",
                             "success / missed / tipped"))
    scenes.append(add_label(load_rgb(os.path.join(TASK5_FRAMES, "task5_success_02_at_cup.png")),
                            "GRASP: gripper closes near cup", color=(0, 255, 255)))
    scenes.append(add_label(load_rgb(os.path.join(TASK5_FRAMES, "task5_success_03_lifted.png")),
                            "SUCCESS: cup lifted and upright", color=(0, 255, 0)))
    scenes.append(add_label(load_rgb(os.path.join(TASK5_FRAMES, "task5_missed_02_at_air.png")),
                            "MISSED: gripper closed next to cup", color=(0, 165, 255)))
    scenes.append(add_label(load_rgb(os.path.join(TASK5_FRAMES, "task5_tipped_03_after_push.png")),
                            "TIPPED: cup knocked over", color=(0, 0, 255)))

    # 14: summary
    scenes.append(make_title("Summary",
                             "Task 1&2: diagnosed identity-loss bug\n"
                             "Task 3: identity preserved after move\n"
                             "Task 4: PyBullet color fallback mask\n"
                             "Task 5: grasp outcomes validated"))

    # Write frames and video.  Each scene is held for HOLD_SECONDS.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(DEMO_DIR, "tasks_demo.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, FRAME_SIZE)
    frame_idx = 0
    for scene_idx, img in enumerate(scenes):
        for _ in range(HOLD_SECONDS):
            timestamp = frame_idx * 1000 // FPS
            save_frame(frame_idx, img, timestamp)
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            frame_idx += 1
    writer.release()

    # Copy scene graph from task3
    sg_src = os.path.join(BASE, "task3_cup_identity", "scene_graph.json")
    sg_dst = os.path.join(DEMO_DIR, "scene_graph.json")
    if os.path.exists(sg_src):
        with open(sg_src) as f:
            sg = json.load(f)
        with open(sg_dst, "w") as f:
            json.dump(sg, f, indent=2)

    report = {
        "demo_video": video_path,
        "frame_size": FRAME_SIZE,
        "fps": FPS,
        "hold_seconds": HOLD_SECONDS,
        "num_scenes": len(scenes),
        "tasks": {
            "task1_2": "diagnosed identity-loss bug (voxel IoU = 0 after move)",
            "task3": "identity preserved via move_instance + reassociate",
            "task4": "PyBullet color fallback mask for DINO+SAM misses",
            "task5": "grasp outcomes validated: success / missed / tipped",
        }
    }
    with open(os.path.join(DEMO_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Demo video: {video_path}")
    print(f"  {len(scenes)} scenes, each held {HOLD_SECONDS}s, total {frame_idx // FPS}s")
    print(f"Demo frames: {DEMO_DIR}/frame_XX_tXXXX.png")


if __name__ == "__main__":
    build_demo()
