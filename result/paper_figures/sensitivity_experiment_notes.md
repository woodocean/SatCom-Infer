# 敏感性实验重绘说明

- 字体：`SimSun`。
- 图均由 `result/runs` 下已有 CSV 重绘，没有重新运行仿真。
- 纵轴统一使用“相对 GS-Only 的归一化时延”，便于跨模型比较。
- 虚线 `y=1` 表示 GS-Only 基线，低于 1 说明该算法优于 GS-Only。
- 为避免 LADP 与贪心、GS-Only 等曲线重合时互相遮挡，图中对不同算法曲线做了极小水平错位，并使用不同线型；该处理只用于视觉区分，不改变原始时延数值。

## 新增图表

- `13_isl_bandwidth_sensitivity_norm.png/pdf`：ISL 带宽敏感性分析。
- `14_gsl_bandwidth_sensitivity_norm.png/pdf`：GSL 带宽敏感性分析。
- `15_node_count_sensitivity_norm.png/pdf`：节点数量敏感性分析。
- `sensitivity_experiment_summary.csv`：上述三类实验的均值摘要。

## 数据源

### ISL 带宽敏感性

- YOLOv5：`result\runs\20260426_010154_isl_yolo_20260426_isl_bw_yolov5_b32_640x640_theory_500to20000_p20_r10_seed42\data\summary_isl_bw_yolov5_b32_640x640_theory_500.0to20000.0_p20.csv`
- ResNet101：`result\runs\20260426_010818_run_20260426_010818_isl_bw_resnet101_b32_224x224_theory_500to20000_p20_r10_seed42\data\summary_isl_bw_resnet101_b32_224x224_theory_500.0to20000.0_p20.csv`
- VGG19：`result\runs\20260426_011345_run_20260426_011345_isl_bw_vgg19_b32_224x224_theory_500to20000_p20_r10_seed42\data\summary_isl_bw_vgg19_b32_224x224_theory_500.0to20000.0_p20.csv`
- Swin-Base：`result\runs\20260426_011827_run_20260426_011827_isl_bw_swin_base_b32_224x224_theory_500to20000_p20_r10_seed42\data\results_long_isl_bw_swin_base_b32_224x224_theory_500.0to20000.0_p20.csv`
- ViT-Huge：`result\runs\20260426_012259_run_20260426_012259_isl_bw_vit_huge_b32_224x224_theory_500to20000_p20_r10_seed42\data\summary_isl_bw_vit_huge_b32_224x224_theory_500.0to20000.0_p20.csv`

### GSL 带宽敏感性

- YOLOv5：`result\runs\20260426_010502_isl_yolo_20260426_gsl_bw_yolov5_b32_640x640_theory_20to200_p20_r10_seed42\data\summary_gsl_bw_yolov5_b32_640x640_theory_20.0to200.0_p21.csv`
- ResNet101：`result\runs\20260426_011130_run_20260426_011130_gsl_bw_resnet101_b32_224x224_theory_20to200_p20_r10_seed42\data\summary_gsl_bw_resnet101_b32_224x224_theory_20.0to200.0_p20.csv`
- VGG19：`result\runs\20260426_011555_run_20260426_011555_gsl_bw_vgg19_b32_224x224_theory_20to200_p20_r10_seed42\data\summary_gsl_bw_vgg19_b32_224x224_theory_20.0to200.0_p20.csv`
- Swin-Base：`result\runs\20260426_012040_run_20260426_012040_gsl_bw_swin_base_b32_224x224_theory_20to200_p20_r10_seed42\data\summary_gsl_bw_swin_base_b32_224x224_theory_20.0to200.0_p20.csv`
- ViT-Huge：`result\runs\20260426_012456_run_20260426_012456_gsl_bw_vit_huge_b32_224x224_theory_20to200_p20_r10_seed42\data\summary_gsl_bw_vit_huge_b32_224x224_theory_20.0to200.0_p20.csv`

### 节点数量敏感性

- YOLOv5：`result\runs\20260426_170935_paper_nodes_yolo_003_node_count_sensitivity_yolov5_b32_640x640_theory_values_p5_r10_seed42\data\summary_node_count_sensitivity_yolov5_b32_640x640_theory_1to5_p5.csv`
- ResNet101：`result\runs\20260426_171034_run_20260426_171034_node_count_sensitivity_resnet101_b32_224x224_theory_values_p5_r10_seed42\data\summary_node_count_sensitivity_resnet101_b32_224x224_theory_1to5_p5.csv`
- VGG19：`result\runs\20260426_171120_run_20260426_171120_node_count_sensitivity_vgg19_b32_224x224_theory_values_p5_r10_seed42\data\summary_node_count_sensitivity_vgg19_b32_224x224_theory_1to5_p5.csv`
- Swin-Base：`result\runs\20260426_171357_run_20260426_171357_node_count_sensitivity_swin_base_b32_224x224_theory_values_p5_r10_seed42\data\summary_node_count_sensitivity_swin_base_b32_224x224_theory_1to5_p5.csv`
- ViT-Huge：`result\runs\20260426_171632_run_20260426_171632_node_count_sensitivity_vit_huge_b32_224x224_theory_values_p5_r10_seed42\data\summary_node_count_sensitivity_vit_huge_b32_224x224_theory_1to5_p5.csv`

## 平均表现最优算法

### ISL 带宽敏感性

| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |
| --- | --- | ---: | ---: |
| YOLOv5 | LADP / 贪心 | 0.393 | 0.359 |
| ResNet101 | LADP | 0.868 | 0.854 |
| VGG19 | LADP | 0.387 | 0.319 |
| Swin-Base | LADP | 0.996 | 0.923 |
| ViT-Huge | GS-Only / LADP / 贪心 | 1.000 | 1.000 |

### GSL 带宽敏感性

| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |
| --- | --- | ---: | ---: |
| YOLOv5 | LADP / 贪心 | 0.359 | 0.338 |
| ResNet101 | LADP | 0.773 | 0.687 |
| VGG19 | LADP | 0.193 | 0.059 |
| Swin-Base | LADP | 0.728 | 0.410 |
| ViT-Huge | GS-Only / LADP / 贪心 | 1.000 | 1.000 |

### 节点数量敏感性

| 模型 | 平均最优算法 | 平均归一化时延 | 最低归一化时延 |
| --- | --- | ---: | ---: |
| YOLOv5 | LADP | 0.360 | 0.352 |
| ResNet101 | LADP | 0.835 | 0.795 |
| VGG19 | LADP | 0.280 | 0.246 |
| Swin-Base | LADP | 0.889 | 0.810 |
| ViT-Huge | GS-Only / LADP / 贪心 | 1.000 | 1.000 |

## 可写进论文的结论

1. ISL 带宽提高后，PMP 模式中需要跨星传输中间特征的算法会受益，但收益不是无限增长；当通信不再是主要瓶颈后，推理计算和分层策略成为主导。
2. GSL 带宽对输入上行和结果回传更敏感，尤其在输入较大的 YOLOv5 场景下更明显；这说明星地链路是端到端时延的重要约束。
3. 节点数量增加会扩大 LADP 的模型切分搜索空间，通常能降低或稳定时延；随机和均匀分配容易引入不必要通信或负载不均，因此波动更大。
4. LADP 在多数敏感性场景下保持较低归一化时延，说明它不是只在单一参数配置下有效，而是对带宽和节点数量变化具有一定鲁棒性。
