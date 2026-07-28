# Xiao : 新增文件，用于本仓库对上游的扩展。
"""Test RoboPercept initialization with progress tracking."""
import sys
sys.path.insert(0, "/home/jx/kitchen/RoboEXP")

import torch
print(f"[1/5] CUDA available: {torch.cuda.is_available()}")

# Direct import to avoid __init__.py loading xarm
print("[2/5] Importing MyGroundingSegment...")
from roboexp.perception.models.grounding_segment import MyGroundingSegment

print("[3/5] Importing MyDenseClip...")
from roboexp.perception.models.dense_clip import MyDenseClip

print("[4/5] Initializing MyGroundingSegment (this may take a while)...")
my_grounding_sam = MyGroundingSegment(device="cuda" if torch.cuda.is_available() else "cpu")
print("  -> MyGroundingSegment done!")

print("[5/5] Initializing MyDenseClip...")
my_dense_clip = MyDenseClip(device="cuda" if torch.cuda.is_available() else "cpu")
print("  -> MyDenseClip done!")

print("\nAll models loaded successfully!")
