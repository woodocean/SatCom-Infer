# FWMS-Feature 权重敏感性实验说明

本实验用于回应“FWMS 特征权重是否合理、模式选择是否过于武断”的问题。

核心思路：不把 FWMS 说成固定权重下必然最优，而是把它定义为可解释的特征加权模式边界框架。不同权重代表不同任务偏好或系统策略。

## 1. 实验方法

本实验不重新运行底层模式评估，而是复用 stage6 已有结果中的：

- PMP 评估结果
- CDP 评估结果
- FWMS-Feature 中记录的 `eta / rho / Bbar` 特征
- Oracle-Min-Latency 上界结果

重新计算不同权重下的 FWMS 选择：

```text
U = w_rho * rho_norm - w_eta * eta_norm + w_bandwidth * Bbar_norm
```

规则：

```text
若 CDP 不可行：选择 PMP 作为保底
若 PMP/CDP 均可行且 U >= 0：选择 PMP
若 PMP/CDP 均可行且 U < 0：选择 CDP
```

## 2. 权重设置

| 权重组 | rho | eta | bandwidth | 含义 |
|---|---:|---:|---:|---|
| equal_weight | 1.0 | 1.0 | 1.0 | 等权，复现实验当前 FWMS-Feature |
| eta_stronger | 1.0 | 2.0 | 1.0 | 增强数据扩张惩罚 |
| bandwidth_weaker | 1.0 | 1.0 | 0.5 | 降低高带宽对 PMP 的奖励 |
| eta_stronger_bandwidth_weaker | 1.0 | 2.0 | 0.5 | 同时增强数据扩张惩罚并降低带宽奖励 |
| eta_dominant | 1.0 | 3.0 | 0.5 | 明显偏重数据扩张边界 |

## 3. 结果文件

汇总目录：

```text
result/mode_selection/fwms_weight_sensitivity_stage6
```

主要产物：

- `fwms_weight_sensitivity_summary.csv`
- `fwms_weight_sensitivity_details.csv`
- `fwms_weight_sensitivity_summary.md`
- `fwms_weight_cdp_selection_ratio.png`
- `fwms_weight_avg_latency.png`

## 4. 关键结果

### 等权 FWMS

等权时，FWMS 明显偏向 PMP：

- YOLO：PMP 100%，CDP 0%
- ResNet101：PMP 92.9%，CDP 7.1%
- Swin-Base：PMP 92.9%，CDP 7.1%
- VGG19 / ViT-Huge：PMP 100%，因为 CDP 不可行

这说明等权 FWMS 不是最低时延选择器。

### 调整权重后

提高 `eta` 权重或降低 `bandwidth` 权重后：

- ResNet101 从主要选择 PMP 转为 100% 选择 CDP。
- Swin-Base 从主要选择 PMP 转为 100% 选择 CDP。
- YOLO 在 `eta_stronger_bandwidth_weaker` 和 `eta_dominant` 下转为 100% 选择 CDP。
- VGG19 / ViT-Huge 始终选择 PMP，因为 CDP 不可行，说明可行性门控优先于特征打分。

## 5. 论文可写结论

1. FWMS 的选择结果对特征权重敏感，说明模式选择不是简单的硬编码规则，而是可根据任务目标进行调节的边界判别框架。
2. 等权策略偏保守，容易受高带宽项影响而偏向 PMP；当增强数据扩张惩罚或降低带宽奖励时，可行的 ResNet101、Swin-Base、YOLO 会更多选择 CDP，并接近 Oracle-Min-Latency 的低时延上界。
3. 对于 VGG19 和 ViT-Huge，CDP 在当前卫星内存条件下不可行，因此无论权重如何，FWMS 都会保底选择 PMP。这说明 FWMS 首先保证任务完成率，再考虑特征偏好。
4. 因此，论文中更稳的表述是：FWMS 不是固定权重的最低时延 Oracle，而是“可行性门控 + 特征加权”的模式边界判别算法。权重可根据任务对时延、能耗、通信压力和内存压力的偏好进行配置。
