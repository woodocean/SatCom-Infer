# 模式选择半实物复刻 smoke 总结

本次半实物复刻不是完整真实卫星网络运行，而是使用 PC/Jetson 的真实推理测量和真实传输测量，再结合 STK 链路带宽、传播时延和算力缩放，复刻理论模式选择实验中的关键趋势。

## 已完成场景

| 场景 | 复刻目的 | 结果文件 |
| --- | --- | --- |
| YOLOv5, batch=64, slot_001 | CDP 可行且时延收益明显时，FWMS 选择 CDP | `result/semi_physical/mode_selection_yolo_stage6_smoke/` |
| VGG19, batch=64, slot_001 | CDP 不可行时，FWMS 回退 PMP | `result/semi_physical/mode_selection_vgg19_stage6_smoke/` |

## 关键结果

| 场景 | 模式 | 理论时延/ms | 半实物时延/ms | 说明 |
| --- | --- | ---: | ---: | --- |
| YOLOv5 | PMP | 11669.57 | 5962.46 | 单路径流水线，半实物下仍慢于 CDP |
| YOLOv5 | CDP | 1842.17 | 2114.21 | 多 worker 数据并行，半实物下保持低时延优势 |
| YOLOv5 | GS-Only | 11766.06 | 15072.35 | 输入回传到 GS 完整推理，通信开销较高 |
| YOLOv5 | FWMS-Feature | 1842.17 | 2114.21 | FWMS 选择 CDP，复刻 CDP 半实物结果 |
| VGG19 | PMP | 3898.85 | 1782.57 | CDP 不可行时的保底模式 |
| VGG19 | GS-Only | 1461.59 | 1782.58 | Oracle 可能选 GS-Only，但 FWMS 仅在 PMP/CDP 边界内判别 |
| VGG19 | FWMS-Feature | 3898.85 | 1782.57 | FWMS 选择 PMP，体现 CDP 不可行时的稳定回退 |

## 可用于论文/汇报的结论

1. 在 YOLOv5 这类 CDP 可行、batch 较大的任务中，半实物复刻结果与理论趋势一致：CDP/FWMS 的端到端时延低于 PMP 和 GS-Only。
2. 在 VGG19 这类完整模型无法部署到单个 worker 的任务中，CDP 不可行，FWMS 回退到 PMP，验证了模式选择算法的可行性门控作用。
3. 半实物平台验证的是趋势和关键开销模型的合理性，不应表述为完全复现真实卫星网络。

## 输出文件

- `semi_physical_mode_results.csv`：逐模式半实物结果明细。
- `semi_physical_summary.csv`：按模式汇总的理论/半实物平均时延。
- `semi_physical_avg_latency_by_mode.png`：半实物平均时延图。
- `semi_physical_theory_vs_real_latency.png`：理论与半实物时延关系图。
