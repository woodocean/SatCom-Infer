# 论文实验图表索引与结论记录

- 绘图字体：`SimSun`。
- 这些图均由已有 CSV 重绘得到，没有重新运行仿真实验。
- 图表顺序按论文叙事排列：PMP 优化、CDP 优化、模式选择与适用边界。
- 全文建议固定使用下表色号，避免同一算法或模式在不同图里换颜色。

## 0. 统一色号

| 对象 | 色号 | 说明 |
| --- | --- | --- |
| LADP / LAWA / PMP | `#2563EB` | 本文核心优化算法或流水线模式 |
| CDP / Greedy | `#E39D2D` | 数据并行模式或贪心算法 |
| GS-Only | `#4A4A4A` | 地面站完整推理基线 |
| Sat-Only | `#8272B2` | 单星完整推理基线 |
| Uniform | `#9CA3AF` | 均匀分配基线 |
| FWMS / GA | `#2A8C88` | 模式选择算法或遗传算法 |
| Random / Oracle | `#E95B45` | 随机基线或最小时延上界 |
| GA | `#059669` | 遗传算法基线 |
| YOLOv5 | `#2563EB` | 模型作为图例对象时使用 |
| ResNet101 | `#F59E0B` | 模型作为图例对象时使用 |
| VGG19 | `#10B981` | 模型作为图例对象时使用 |
| Swin-Base | `#7C3AED` | 模型作为图例对象时使用 |
| ViT-Huge | `#EF4444` | 模型作为图例对象时使用 |

## 1. PMP 优化实验

- `01_pmp_algorithm_latency_norm.png/pdf`：PMP 模式下 LADP 与 Greedy、GA、Random、Uniform、GS-Only 的归一化时延对比。
- `02_pmp_algorithm_energy.png/pdf`：PMP 模式下不同算法的星载能耗对比。

| 模型 | PMP时延最优算法 | 最小归一化时延 | PMP能耗最优算法 | 最小星载能耗(J) |
| --- | --- | --- | --- | --- |
| YOLOv5 | LADP | 0.368 | LADP | 31.49 |
| ResNet101 | LADP | 0.859 | LADP | 9.72 |
| VGG19 | LADP | 0.342 | LADP | 3.81 |
| Swin-Base | LADP | 0.906 | LADP | 10.34 |
| ViT-Huge | LADP | 1.000 | LADP | 10.95 |

## 2. CDP 优化实验

- `03_cdp_algorithm_latency.png/pdf`：CDP 模式下 LAWA 与数据分配基线算法的平均时延对比。
- `04_cdp_algorithm_latency_norm.png/pdf`：CDP 模式下各算法相对 Sat-Only 基线的归一化时延。

| CDP最优算法 | 平均时延(ms) | 相对Sat-Only比例 |
| --- | --- | --- |
| LAWA | 45.23 | 0.176 |

## 3. 模式选择与适用边界

- `05_mode_latency_by_model.png/pdf`：PMP、CDP、GS-Only、Sat-Only、FWMS 的跨模型时延对比。
- `06_mode_energy_by_model.png/pdf`：不同模式的星载能耗对比。
- `07_mode_completion_by_model.png/pdf`：不同模式在 STK 动态时间片下的任务完成率。
- `08_fwms_oracle_selection_distribution.png/pdf`：FWMS 与最小时延 Oracle 的选择分布差异。
- `09_batch_cdp_feasibility.png/pdf`：CDP 在不同模型和 batch 下的可行性边界。
- `10_batch_oracle_selection_distribution.png/pdf`：不同 batch 下最小时延模式的分布。
- `11_cdp_boundary_sensitivity.png/pdf`：CDP 对 batch 和 worker 数量的敏感性。
- `12_fixed_mode_fwms_completion.png/pdf`：固定模式与 FWMS 的任务完成率对比。

| 模型 | PMP完成率 | CDP完成率 | 边界结论 |
| --- | --- | --- | --- |
| YOLOv5 | 100% | 100% | CDP可行时低时延占优 |
| ResNet101 | 100% | 100% | CDP可行时低时延占优 |
| VGG19 | 100% | 0% | CDP不可行，PMP/GS-Only承担保底 |
| Swin-Base | 100% | 100% | CDP可行时低时延占优 |
| ViT-Huge | 100% | 0% | CDP不可行，PMP/GS-Only承担保底 |

## 3.5 敏感性实验

- `13_isl_bandwidth_sensitivity_norm.png/pdf`：ISL 带宽敏感性分析。
- `14_gsl_bandwidth_sensitivity_norm.png/pdf`：GSL 带宽敏感性分析。
- `15_node_count_sensitivity_norm.png/pdf`：节点数量敏感性分析。
- `16_cdp_energy_sensitivity.png/pdf`：CDP 模式在不同 batch 和 worker 数下的星载能耗敏感性分析。
- `sensitivity_experiment_notes.md`：敏感性实验数据源、最优算法摘要和可写进论文的结论。
- `cdp_energy_sensitivity_notes.md`：CDP 星载能耗敏感性图的数据源和结论。
- `runs_experiment_inventory.md`：`result/runs` 下已有实验资产索引。

## 4. 汇报用结论

- PMP 内部优化结论：LADP 在 STK 动态拓扑下通常取得更低的归一化时延，并能降低星载能耗，说明模型切分优化是有效的。
- CDP 内部优化结论：LAWA 在数据并行场景下明显优于 Sat-Only、均匀分配和贪心等基线，说明数据分配优化是有效的。
- 模式边界结论：CDP 在可行且 batch 较大时低时延优势明显，但对单星内存和 worker 可见性更敏感；PMP 更稳定，适合作为大模型、资源受限或 CDP 不可行时的保底模式。
- FWMS 叙事结论：FWMS 不应被表述为最小时延 Oracle，而应表述为结合模型特征、通信特征、内存约束和资源状态的模式边界判别方法。
