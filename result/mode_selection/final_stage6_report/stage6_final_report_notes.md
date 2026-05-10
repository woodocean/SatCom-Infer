# Stage6 模式选择最终汇总说明

本目录由 `plot_stage6_mode_selection_report.py` 生成，输入为五个 `mode_selection_*_stage6_feature_oracle_b64` 结果目录。

## 关键口径

- `FWMS-Feature`：论文算法叙事使用的特征加权模式边界判别器，先做可行性筛选，再根据任务/资源特征在 PMP 与 CDP 间判别。
- `Oracle-Min-Latency`：离线理论上界基线，已知所有候选模式预测时延后选择最低者，不应表述为在线 FWMS。
- `feasible_rate`：该模式在 42 个 STK 时间片中的可行率，也可作为完成率使用。

## 可直接引用的结论

- CDP 在 YOLOv5、ResNet101、Swin-Base 上可行且时延最低，说明批量数据并行在资源充足时有明显优势。
- VGG19 与 ViT-Huge 下 CDP/Sat-Only 不可行，主要体现完整模型部署的内存边界，此时 PMP 或 GS-Only 是稳定完成任务的保底路径。
- FWMS-Feature 的价值不等同于 Oracle 的最低时延，而是把固定模式扩展为基于可行性和特征的稳定模式判别。

## 主要输出

- `stage6_mode_summary.csv/md`：跨模型、跨模式平均时延、能耗和完成率。
- `stage6_selector_distribution.csv/md`：FWMS-Feature 与 Oracle-Min-Latency 的选择分布。
- `stage6_fwms_oracle_gap.csv/md`：FWMS-Feature 相对 Oracle-Min-Latency 的时延差距和边界解释。
- `stage6_avg_latency_by_model.png/pdf`：平均时延对比图。
- `stage6_avg_energy_by_model.png/pdf`：平均卫星能耗对比图。
- `stage6_completion_heatmap.png/pdf`：可行率/完成率热力图。
- `stage6_selector_distribution.png/pdf`：两类选择器的模式选择分布。
- `stage6_fwms_oracle_latency_gap.png/pdf`：FWMS 与 Oracle 的时延比。

## 表 1：跨模型模式摘要

| 模型 | 模式 | 完成率 | 平均时延(ms) | 平均卫星能耗(J) |
| --- | --- | --- | --- | --- |
| YOLOv5 | PMP | 100% | 6597.55 | 63.06 |
| YOLOv5 | CDP | 100% | 1928.02 | 74.09 |
| YOLOv5 | GS-Only | 100% | 12134.28 | 119.04 |
| YOLOv5 | Sat-Only | 100% | 7435.22 | 73.55 |
| YOLOv5 | FWMS-Feature | 100% | 1928.02 | 74.09 |
| YOLOv5 | Oracle-Min-Latency | 100% | 1928.02 | 74.09 |
| ResNet101 | PMP | 100% | 1957.53 | 20.20 |
| ResNet101 | CDP | 100% | 380.45 | 14.80 |
| ResNet101 | GS-Only | 100% | 1530.60 | 14.58 |
| ResNet101 | Sat-Only | 100% | 1370.97 | 14.73 |
| ResNet101 | FWMS-Feature | 100% | 380.45 | 14.80 |
| ResNet101 | Oracle-Min-Latency | 100% | 380.45 | 14.80 |
| VGG19 | PMP | 100% | 1747.12 | 18.63 |
| VGG19 | CDP | 0% | - | - |
| VGG19 | GS-Only | 100% | 1534.05 | 14.58 |
| VGG19 | Sat-Only | 0% | - | - |
| VGG19 | FWMS-Feature | 100% | 1747.12 | 18.63 |
| VGG19 | Oracle-Min-Latency | 100% | 1343.92 | 14.00 |
| Swin-Base | PMP | 100% | 1931.36 | 20.95 |
| Swin-Base | CDP | 100% | 514.34 | 25.05 |
| Swin-Base | GS-Only | 100% | 1558.18 | 14.58 |
| Swin-Base | Sat-Only | 100% | 1849.05 | 24.17 |
| Swin-Base | FWMS-Feature | 100% | 709.08 | 22.19 |
| Swin-Base | Oracle-Min-Latency | 100% | 514.34 | 25.05 |
| ViT-Huge | PMP | 100% | 2378.16 | 21.91 |
| ViT-Huge | CDP | 0% | - | - |
| ViT-Huge | GS-Only | 100% | 1627.23 | 14.58 |
| ViT-Huge | Sat-Only | 0% | - | - |
| ViT-Huge | FWMS-Feature | 100% | 2378.16 | 21.91 |
| ViT-Huge | Oracle-Min-Latency | 100% | 1627.23 | 14.58 |

## 表 2：FWMS 与 Oracle 差距

| 模型 | PMP完成率 | CDP完成率 | FWMS平均时延(ms) | Oracle平均时延(ms) | FWMS/Oracle时延比 | 边界解释 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | 100% | 100% | 1928.02 | 1928.02 | 1.00x | CDP可行且低时延优势明显，适合批量数据并行。 |
| ResNet101 | 100% | 100% | 380.45 | 380.45 | 1.00x | CDP可行且低时延优势明显，适合批量数据并行。 |
| VGG19 | 100% | 0% | 1747.12 | 1343.92 | 1.30x | CDP因完整模型部署约束不可行，PMP/GS-Only承担保底。 |
| Swin-Base | 100% | 100% | 709.08 | 514.34 | 1.38x | CDP可行且低时延优势明显，适合批量数据并行。 |
| ViT-Huge | 100% | 0% | 2378.16 | 1627.23 | 1.46x | CDP因完整模型部署约束不可行，PMP/GS-Only承担保底。 |
