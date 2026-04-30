# Stage6 Batch 边界补充实验说明

本实验补跑五个模型在 `batch=32` 下的 Stage6 模式选择，并与已有 `batch=64` 结果对比。目的不是替代主实验，而是补充说明任务规模变化时 CDP/PMP/GS-Only 的适用边界。

## 新增实验

- `mode_selection_yolo_stage6_feature_oracle_b32`
- `mode_selection_resnet101_stage6_feature_oracle_b32`
- `mode_selection_vgg19_stage6_feature_oracle_b32`
- `mode_selection_swin_base_stage6_feature_oracle_b32`
- `mode_selection_vit_huge_stage6_feature_oracle_b32`

## 主要结论

- YOLOv5、ResNet101、Swin-Base 在 batch=32/64 下 CDP 都可行，说明这些模型在当前资源条件下适合数据并行。
- VGG19 在 batch=32/64 下 CDP 都不可行，说明其瓶颈主要来自完整模型部署约束，而不是 batch 规模。
- ViT-Huge 在 batch=32 下 CDP 可行，但 batch=64 下 CDP 不可行，说明任务规模会改变模式可行边界。
- batch=32 下 ViT-Huge 虽然 CDP 可行，但 Oracle 仍选择 GS-Only，说明“可行”不等于“最低时延”，这能支撑 FWMS 需要综合特征而不是只做内存筛选。

## 输出文件

- `batch_boundary_mode_summary.csv/md`：batch=32/64 的跨模型模式摘要。
- `batch_boundary_selector_distribution.csv/md`：选择器分布。
- `batch_boundary_notes.md`：每个模型和 batch 的边界观察表。
- `batch_boundary_cdp_completion.png/pdf`：CDP 可行率随 batch 的变化。
- `batch_boundary_oracle_selection.png/pdf`：Oracle 选择模式随 batch 的变化。
- `batch_boundary_fwms_oracle_latency_ratio.png/pdf`：FWMS 相对 Oracle 的时延比变化。

## 边界观察表

| 模型 | batch | CDP完成率 | PMP平均时延(ms) | CDP平均时延(ms) | Oracle平均时延(ms) | Oracle选择分布 | 边界观察 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 32 | 100% | 3315.54 | 1002.02 | 1002.02 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| YOLOv5 | 64 | 100% | 6597.55 | 1995.62 | 1995.62 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| ResNet101 | 32 | 100% | 961.23 | 216.99 | 216.99 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| ResNet101 | 64 | 100% | 1957.53 | 453.03 | 453.03 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| VGG19 | 32 | 0% | 465.28 | - | 459.14 | GS-Only:5%, PMP:95% | 完整模型无法部署到可见 worker，CDP 不可行。 |
| VGG19 | 64 | 0% | 1747.12 | - | 1343.92 | GS-Only:69%, PMP:31% | 完整模型无法部署到可见 worker，CDP 不可行。 |
| Swin-Base | 32 | 100% | 981.70 | 265.32 | 265.32 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| Swin-Base | 64 | 100% | 1931.36 | 482.25 | 482.25 | CDP:100% | CDP 可行且通常成为时延上界选择，体现数据并行优势。 |
| ViT-Huge | 32 | 100% | 1209.53 | 2962.42 | 834.53 | GS-Only:100% | batch=32 时 CDP 可行但不一定最低时延，说明可行性不等于最优性。 |
| ViT-Huge | 64 | 0% | 2378.16 | - | 1627.23 | GS-Only:100% | 完整模型无法部署到可见 worker，CDP 不可行。 |
