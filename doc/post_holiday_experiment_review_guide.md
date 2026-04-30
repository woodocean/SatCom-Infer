# 五一后实验 Review 指南

这份文档用于假期后快速恢复上下文。当前阶段不再把 FWMS 叙述成“最低时延选择器”，而是叙述为“基于可行性与任务特征的模式边界判别方法”。`Oracle-Min-Latency` 只作为离线理论上界，用来说明如果已知所有模式预测结果，最低时延能达到什么水平。

## 1. 已完成的新增实验

### 1.1 Stage6 五模型主实验

输出目录：

```text
result/mode_selection/final_stage6_report/
```

输入数据来自五个 `batch=64` 的完整 Stage6 结果：

```text
result/mode_selection/mode_selection_yolo_stage6_feature_oracle_b64/
result/mode_selection/mode_selection_resnet101_stage6_feature_oracle_b64/
result/mode_selection/mode_selection_vgg19_stage6_feature_oracle_b64/
result/mode_selection/mode_selection_swin_base_stage6_feature_oracle_b64/
result/mode_selection/mode_selection_vit_huge_stage6_feature_oracle_b64/
```

主要产物：

```text
stage6_mode_summary.md
stage6_selector_distribution.md
stage6_fwms_oracle_gap.md
stage6_final_report_notes.md
stage6_avg_latency_by_model.png
stage6_avg_energy_by_model.png
stage6_completion_heatmap.png
stage6_selector_distribution.png
stage6_fwms_oracle_latency_gap.png
```

可用结论：

- CDP 在 YOLOv5、ResNet101、Swin-Base 上可行且时延最低，说明资源充足时数据并行有明显低时延优势。
- VGG19 和 ViT-Huge 在 `batch=64` 下 CDP/Sat-Only 不可行，说明完整模型部署约束会形成明确的模式边界。
- FWMS-Feature 保证所有时间片可完成，但它不是 Oracle；Oracle-Min-Latency 是离线上界，不应包装成真实在线算法。

### 1.2 Batch 边界补充实验

输出目录：

```text
result/mode_selection/batch_boundary_stage6/
```

本次新补跑了五个模型的 `batch=32` Stage6 结果，并与已有 `batch=64` 结果对比：

```text
result/mode_selection/mode_selection_yolo_stage6_feature_oracle_b32/
result/mode_selection/mode_selection_resnet101_stage6_feature_oracle_b32/
result/mode_selection/mode_selection_vgg19_stage6_feature_oracle_b32/
result/mode_selection/mode_selection_swin_base_stage6_feature_oracle_b32/
result/mode_selection/mode_selection_vit_huge_stage6_feature_oracle_b32/
```

主要产物：

```text
batch_boundary_mode_summary.md
batch_boundary_selector_distribution.md
batch_boundary_notes.md
batch_boundary_report_notes.md
batch_boundary_cdp_completion.png
batch_boundary_oracle_selection.png
batch_boundary_fwms_oracle_latency_ratio.png
```

可用结论：

- YOLOv5、ResNet101、Swin-Base 在 `batch=32/64` 下 CDP 都可行，适合数据并行。
- VGG19 在 `batch=32/64` 下 CDP 都不可行，说明其瓶颈主要是完整模型部署，而不是 batch 规模。
- ViT-Huge 在 `batch=32` 下 CDP 可行，但 `batch=64` 下 CDP 不可行，说明任务规模会改变模式可行边界。
- ViT-Huge 在 `batch=32` 下虽然 CDP 可行，但 Oracle 仍选择 GS-Only，说明“可行”不等于“最低时延”，这能支撑 FWMS 不能只做内存筛选。

### 1.3 CDP 敏感性实验

输出目录：

```text
result/mode_selection/cdp_sensitivity_yolo_stage6/
doc/cdp_sensitivity_experiment_summary.md
```

覆盖内容：

- `batch=16/32/64/128`，固定最多 4 个 worker。
- `worker=2/3/4`，固定 `batch=64`。

可用结论：

- CDP 随 batch 增大体现并行收益，但过大 batch 会触发资源边界。
- worker 数增加时，CDP 平均时延下降，说明 LAWA 的多 worker 分配确实有作用。

### 1.4 FWMS 权重敏感性实验

输出目录：

```text
result/mode_selection/fwms_weight_sensitivity_stage6/
doc/fwms_weight_sensitivity_experiment_summary.md
```

覆盖内容：

- equal weight。
- 增强计算异构特征权重。
- 减弱通信带宽保守项。
- 组合调整。

可用结论：

- FWMS-Feature 的选择边界会随特征权重移动，不是单纯“内存过不了就 PMP”。
- VGG19、ViT-Huge 在 CDP 不可行时不会因权重变化强行选择 CDP，说明可行性门控仍然有效。
- YOLOv5、ResNet101、Swin-Base 在特征权重调整后会更多转向 CDP，说明特征权重确实能表达模式偏好。

## 2. 当前最稳的论文叙事

推荐表述：

```text
本文不是把 PMP 与 CDP 简化为同一个最低时延优化问题，而是将二者视为两类具有不同适用边界的协作推理模式。CDP 在模型可完整部署、batch 较大且多颗 worker 可见时具有明显低时延优势；PMP 在完整模型无法部署或资源受限时提供模型切分执行路径，提升系统完成率与鲁棒性。FWMS 通过可行性筛选和任务特征加权，在不同模型、batch 和资源条件下判别适用模式，从而避免固定模式在边界场景下失效。
```

不要再主张：

```text
FWMS 总能选择最低时延模式。
```

可以主张：

```text
FWMS 不是 Oracle，而是可解释、低复杂度、面向稳定性的模式边界判别算法。
```

## 3. 五一后优先 Review 顺序

1. 先看 `result/mode_selection/final_stage6_report/stage6_final_report_notes.md`。
2. 再看 `result/mode_selection/batch_boundary_stage6/batch_boundary_report_notes.md`。
3. 然后看两组图：`stage6_completion_heatmap.png` 和 `batch_boundary_cdp_completion.png`。
4. 最后决定论文是否采用 FWMS 权重敏感性图；如果篇幅紧，可以只放结论不放图。

## 4. 后续还缺什么

最值得补的是 Jetson 半实物验证，但不建议全量铺开。建议五一后只做代表性实验：

- YOLOv5：展示 CDP 可行且低时延。
- VGG19 或 ViT-Huge：展示 CDP 受完整模型部署约束影响，PMP/GS-Only 作为保底。
- batch 只选 `32/64`。
- 模式只选 `PMP / CDP / GS-Only`。

不建议继续做：

- 新增大模型。
- 新增强化学习/神经网络版 FWMS。
- 为了让 PMP 赢而修改 PMP/CDP 定义。
- 全量扫所有带宽、节点数、batch、模型组合。

## 5. 复现实验命令

生成 Stage6 主实验汇总：

```powershell
python plot_stage6_mode_selection_report.py --input-root result/mode_selection --output-dir result/mode_selection/final_stage6_report
```

生成 batch 边界汇总：

```powershell
python plot_stage6_batch_boundary_report.py --input-root result/mode_selection --output-dir result/mode_selection/batch_boundary_stage6
```

单独重跑一个 batch=32 模式选择实验示例：

```powershell
python mode_selection_experiment.py --stk-run-dir result/stk_dynamic/stk_dynamic_yolo_001 --run-id mode_selection_yolo_stage6_feature_oracle_b32 --batch-size-override 32 --cdp-max-workers 4
```
