# 模式选择：边界与稳定性叙事

本文档用于把当前模式选择实验从“最低时延选择器”叙事，调整为“可行性与任务特征驱动的模式边界判别”叙事。

## 1. 两个选择器必须区分

### FWMS-Feature

`FWMS-Feature` 对应论文中的特征加权模式选择思想。它不是直接枚举所有模式后选最低时延，而是先做可行性筛选，再根据模型特征、通信特征和资源状态判断更适合 PMP 还是 CDP。

当前实现：

```text
U = w_rho * rho_norm - w_eta * eta_norm + w_bandwidth * Bbar_norm
```

含义：
- `eta`：中间特征/输入数据的数据扩张程度。越大，越不利于 PMP。
- `rho`：CDP worker 算力异构程度。越大，越不利于 CDP。
- `Bbar`：PMP 路径平均带宽。越大，越有利于 PMP。

当前版本权重为等权，只用于完成结构性区分和边界分析；如果后续要把它作为性能最优选择器，需要重新校准权重。

### Oracle-Min-Latency

`Oracle-Min-Latency` 是预测最小时延上界基线。它先完整评估 `PMP / GS-Only / Sat-Only / CDP`，再从所有可行模式中选择预测时延最低者。

它的作用不是模拟真实在线算法，而是回答：

```text
如果每个 slot 里所有模式都已经被完整评估，最低时延上界落在哪个模式？
```

因此论文中应把它写成 baseline/upper bound，而不是 FWMS 本体。

## 2. 当前跨模型结果

数据来自：

```text
result/mode_selection/mode_selection_yolo_stage6_feature_oracle_b64
result/mode_selection/mode_selection_resnet101_stage6_feature_oracle_b64
result/mode_selection/mode_selection_vgg19_stage6_feature_oracle_b64
result/mode_selection/mode_selection_vit_huge_stage6_feature_oracle_b64
result/mode_selection/mode_selection_swin_base_stage6_feature_oracle_b64
```

每个模型 42 个 STK 动态时间片，batch size 为 64。

| 模型 | CDP 可行率 | Sat-Only 可行率 | Oracle 主要选择 | FWMS-Feature 主要选择 | 结论 |
|---|---:|---:|---|---|---|
| YOLO | 100% | 100% | CDP 42/42 | PMP 42/42 | CDP 可行时低时延优势明显，当前等权 FWMS 偏保守。 |
| ResNet101 | 100% | 100% | CDP 42/42 | PMP 39/42 | CDP 低时延上界稳定，FWMS 权重仍需校准。 |
| VGG19 | 0% | 0% | GS-Only 29/42，PMP 13/42 | PMP 42/42 | CDP 因完整模型内存压力失效，模式选择体现保底价值。 |
| ViT-Huge | 0% | 0% | GS-Only 42/42 | PMP 42/42 | 大模型完整上星不可行，GS/PMP 是可行路径。 |
| Swin-Base | 100% | 100% | CDP 42/42 | PMP 39/42 | CDP 可行时低时延优势明显，FWMS 等权偏向 PMP。 |

更完整的表格见：

```text
result/mode_selection/cross_model_stage6_feature_oracle/mode_feasibility_completion_table.md
result/mode_selection/cross_model_stage6_feature_oracle/mode_boundary_conclusion_table.md
```

## 3. 论文叙事建议

不建议写：

```text
FWMS 能自动选择最低时延模式。
```

更稳的写法是：

```text
FWMS 通过可行性约束和任务特征判断 PMP/CDP/GS-Only 等模式的适用边界，避免固定模式在动态拓扑和异构资源下失效，从而提升系统完成率和稳定性。
```

这样能解释三个现象：
- CDP 可行时，它在当前理论模型中经常是最低时延模式。
- CDP 不可行时，固定 CDP 会失败，而 FWMS 可以切换到 PMP 或 GS-Only。
- PMP 不必强行证明在所有低时延场景中胜过 CDP，它的价值是模型切分和保底可行性。

## 4. 当前仍需注意的风险

- 当前 `FWMS-Feature` 使用等权特征，跨模型结果显示它偏向 PMP；这不能作为“FWMS 已经达到低时延最优”的证据。
- 如果论文保留原始 FWMS 公式，需要把它解释为启发式边界判别，而不是 Oracle。
- 如果想增强说服力，后续最值得补的是权重敏感性或规则消融：例如展示调整 `w_eta / w_bandwidth` 后，FWMS 如何从保守 PMP 策略逐步靠近 CDP 低时延上界。
