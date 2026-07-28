# 测试与记录文件归档

本目录只存放用户新增/测试/记录性质的文件。原始仓库文件、子模块、核心代码和脚本均不在此列。

## 目录结构（按先后/逻辑顺序）

- `01_phase0_mapping/`：phase0 建图相关视频、图片、笔记
- `02_perception_tests/`：perception 测试图（clip、grounding、camera 参数）
- `03_pr2_tests/`：PR2 机器人测试图（home views、motion sequences、head views）
- `04_tamp_records/`：TAMP 生成记录与统计
- `05_experiments/`：cup / pick-place / dynamic demo 实验输出
  - `01_demo_video/`
  - `02_dynamic_cup_franka/`
  - `03_dynamic_cup_pr2/`
  - `04_tamp_pick_place_pr2/`
  - `05_cup_identity_tracking/`（原 task3）
  - `06_cup_grasp_validation/`（原 task5）
  - `07_perception_parameter_sweep/`（原 perception_archive）
- `06_scene_graph_records/`：scene graph 记录
- `07_misc/`：杂项记录

## 命名规则

- 顶层模块和子任务用两位序号前缀表示顺序。
- 每个文件夹内文件已有的 `01_`、`02_` 前缀表示时间顺序，文件名已去掉原始时间戳。
