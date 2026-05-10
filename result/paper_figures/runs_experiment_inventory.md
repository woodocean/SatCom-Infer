# `result/runs` 实验资产索引

这份索引只整理已有实验结果，不代表重新运行实验。

## 可直接服务论文主线的结果

| 实验类型 | 覆盖模型 | 数据位置 | 当前图表 |
| --- | --- | --- | --- |
| PMP 算法对比 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/stk_dynamic/cross_model/stk_cross_model_summary_long.csv` 与早期 `algo_theory_*` run | `01_pmp_algorithm_latency_norm.*` |
| PMP 星载能耗对比 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/runs/*energy_comparison*/data/summary_*.csv` | `02_pmp_algorithm_energy.*` |
| CDP 算法对比 | YOLOv5 等模式选择链路结果 | `result/mode_selection/*/data/slot_mode_results.csv` | `03_cdp_algorithm_latency.*`、`04_cdp_algorithm_latency_norm.*` |
| ISL 带宽敏感性 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/runs/*isl_bw*/data/*.csv` | `13_isl_bandwidth_sensitivity_norm.*` |
| GSL 带宽敏感性 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/runs/*gsl_bw*/data/*.csv` | `14_gsl_bandwidth_sensitivity_norm.*` |
| 节点数量敏感性 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/runs/*node_count_sensitivity*/data/*.csv` | `15_node_count_sensitivity_norm.*` |
| STK 动态拓扑 PMP 评估 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/runs/*stk_dynamic_pmp*/data/*.csv` 与 `result/stk_dynamic/` | 已汇总进跨模型 PMP 图表 |
| 模式选择与边界 | YOLOv5、ResNet101、VGG19、Swin-Base、ViT-Huge | `result/mode_selection/` | `05` 到 `12` 号图 |

## 历史或辅助结果

| 类型 | 说明 | 建议 |
| --- | --- | --- |
| `result/v4.0/*.png` | 早期导出的英文/旧风格图片，包含 ISL/GSL 带宽和理论-实物对比 | 不建议直接放论文，已用统一风格重画 |
| `20260425_*isl_bw_yolov5*` | YOLOv5 ISL 带宽早期 run | 已选择更新的 `20260426_010154_*` 作为论文图数据源 |
| 多个 `paper_nodes_yolo_*` | YOLOv5 节点数量敏感性多次补跑 | 已选择最新的 `20260426_170935_*` 作为论文图数据源 |
| `physical_pmp_*` | 一键半实物 PMP 尝试记录，部分卡在 ACK/流控或只完成局部链路 | 暂不作为论文主图，只可作为平台调试记录 |

## 当前重绘输出

- `13_isl_bandwidth_sensitivity_norm.png/pdf`：ISL 带宽敏感性，中文标签，统一算法配色。
- `14_gsl_bandwidth_sensitivity_norm.png/pdf`：GSL 带宽敏感性，中文标签，统一算法配色。
- `15_node_count_sensitivity_norm.png/pdf`：节点数量敏感性，中文标签，统一算法配色。
- `sensitivity_experiment_notes.md`：上述三类敏感性实验的数据源和论文结论。
- `sensitivity_experiment_summary.csv`：三类敏感性实验按模型和算法汇总的归一化时延统计。
