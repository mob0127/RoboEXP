# RoboEXP 动态物体跟踪补丁 —— 工作日志

**日期**: 2026-07-23
**目标**: 在现有 RoboEXP 上打补丁，使其能够识别“被移动过的同一物体”，并支持场景图的增量更新与基本的抓取验证。

---

## 背景与决策

- RoboEXP 原生的 `_merge_scene` 只通过 **voxel IoU + CLIP 相似度** 匹配实例。物体移动后无重叠，会被当成新实例。
- Scene Graph 默认只能初始化一次（`scene_graph_option=None` 时断言 `action_scene_graph is None`），没有通用移动更新机制。
- 本次补丁采用**语义标签 + 3D 中心距离**进行重关联，不引入 PyBullet body_id，以保持对真实相机的潜在适用性。
- 先实现核心身份保持（任务 1-3），再补感知 fallback 与抓取验证。

---

## 任务列表

### P0: 核心身份保持

- [x] **任务 1**: 在 `RoboMemory._merge_scene` 中增加位置重关联逻辑
  - 对跨时间步的 memory merge 启用；
  - 当 IoU 匹配失败但同标签实例中心距离 < 阈值时，强制关联为同一实例。

- [x] **任务 2**: 给 `myInstance` 增加 `move_instance`
  - 物体明显移动时，用新观测的体素替换旧体素，保留 `instance_id`；
  - 避免 `merge_instance` 的 `union` 导致实例同时出现在原处和新处。

- [x] **任务 3**: 新增 `scene_graph_option={"type": "reassociate"}`
  - 允许在 scene graph 已存在时更新已有节点；
  - 更新节点 instance、重新计算 parent relation（如 `on` 哪个 surface）。

### P1: 感知补强与抓取验证

- [ ] **任务 4**: 给 `RoboPercept` 增加 fallback 机制
  - DINO+SAM 失效时，可用颜色阈值或 SAM auto + CLIP 生成 mask。

- [ ] **任务 5**: 增加抓取前后验证
  - 检测“没抓上”、“弄翻”、“成功”三类结果。

### P2: 可选

- [ ] **任务 6**: 坐标系一致性修复（待定，影响面大）

---

## 修改文件

1. `RoboEXP/roboexp/memory/instance.py`
   - 新增 `myInstance.move_instance(instance)` 方法。

2. `RoboEXP/roboexp/memory/robo_memory.py`
   - `RoboMemory.__init__` 增加动态跟踪参数；
   - `_merge_scene` 增加 `allow_position_association` 与两轮匹配（IoU → 位置）；
   - `_update_with_current_observations` 调用 `_merge_scene` 时启用位置关联；
   - 新增 `_find_parent_surface` 辅助函数；
   - `_update_scene_graph` 增加 `"reassociate"` 分支，支持更新已有节点和添加新节点；

3. `RoboEXP/roboexp/memory/scene_graph/node.py`
   - `ObjectNode` 新增 `update_parent(new_parent, parent_relation)` 方法。

4. `RoboEXP/experiments/cup_move_scene_graph.py`
   - 改为单 `RoboMemory` + 单环境；
   - 移动后用 `scene_graph_option={"type": "reassociate"}` 更新场景图；
   - 启用 `position_association_enabled=True`。

---

## 进度记录

### 2026-07-23

- 用户确认任务 1 使用语义标签，不加 body_id 增强。
- 完成任务 1-3 的代码修改。
- 运行 `cup_move_scene_graph.py` 验证。

### 2026-07-25

- 运行验证成功：
  - 移动前节点：`cup_0` / `cup_0_instance`，中心 `[2.22, -0.28, 1.06]`；
  - 移动后节点：**仍为** `cup_0` / `cup_0_instance`，中心更新为 `[2.11, 0.40, 1.50]`；
  - `identity_preserved: true`。
- 修复了实验过程中 `kitchen-worlds` 子模块（`pybullet_planning`、`pddlstream`、`assets/models`、`lisdf`）意外为空的问题，从 `kitchen-roboexp` 备份复制恢复。

---

## 验证产物

- 视频：`/home/jx/kitchen/RoboEXP/experiments/outputs/cup_move_scene_graph.mp4`
- 报告：`/home/jx/kitchen/RoboEXP/experiments/outputs/cup_move_report.json`

关键结果：

```json
{
  "num_cup_nodes_before": 1,
  "num_cup_nodes_after": 1,
  "identity_preserved": true
}
```

---

## 下一步

- 任务 4：给 `RoboPercept` 加 fallback，摆脱实验中手动颜色 mask 的限制。
- 任务 5：在 `robo_act` 或适配层中增加抓取验证逻辑。
