# CDP 星载能耗敏感性图

- 数据源：`result\mode_selection\cdp_sensitivity_yolo_stage6\cdp_sensitivity_summary.csv`。
- `16_cdp_energy_sensitivity.png/pdf`：CDP 在不同 batch 和 worker 数下的平均星载能耗。
- batch=128 时 CDP 不可行，因此图中标注为“不可行”，不填充能耗数值。

## 可写进论文的结论

- 随 batch 从 16 增加到 64，CDP 平均星载能耗从约 18.89 J 增加到 77.68 J，说明任务规模增大时并行推理的星上资源消耗同步上升。
- 在 batch=64 下，worker 上限从 2 增加到 4 时，平均星载能耗变化不大，但平均时延显著下降；这说明 CDP 的主要收益来自多 worker 并行缩短完成时间，而不是降低总星载能耗。
- batch=128 下 CDP 不可行，说明 CDP 的低时延优势存在资源边界，固定使用 CDP 可能导致任务失败。