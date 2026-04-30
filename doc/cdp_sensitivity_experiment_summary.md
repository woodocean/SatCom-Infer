# CDP 敏感性实验说明

本实验用于补强模式选择中的 CDP 适用边界结论：CDP 并不是无条件最优，而是依赖 batch、worker 数量和完整模型可装载性。

## 1. 实验设置

模型：`YOLO`

场景：STK 动态拓扑，来自：

```text
result/stk_dynamic/stk_dynamic_yolo_001
```

每组实验使用 42 个有效时间片。CDP 定义为无聚合器版本：

```text
RS 分发 batch -> 多个 worker 独立完整推理 -> 各 worker 直接回传 GS
```

## 2. 已完成实验矩阵

Batch 敏感性：

```powershell
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_cdp_sens_batch_b16_w4 --batch-size-override 16 --cdp-max-workers 4
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_cdp_sens_batch_b32_w4 --batch-size-override 32 --cdp-max-workers 4
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_stage6_feature_oracle_b64 --batch-size-override 64 --cdp-max-workers 4
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_cdp_sens_batch_b128_w4 --batch-size-override 128 --cdp-max-workers 4
```

Worker 数敏感性：

```powershell
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_cdp_sens_batch_b64_w2 --batch-size-override 64 --cdp-max-workers 2
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_cdp_sens_batch_b64_w3 --batch-size-override 64 --cdp-max-workers 3
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_stage6_feature_oracle_b64 --batch-size-override 64 --cdp-max-workers 4
```

说明：当前 profile 数据库没有 YOLO 的 `b8_640x640`，因此本轮未跑 batch=8。

## 3. 结果文件

汇总目录：

```text
result/mode_selection/cdp_sensitivity_yolo_stage6
```

主要产物：

- `cdp_sensitivity_summary.csv`
- `cdp_sensitivity_summary.md`
- `cdp_batch_sensitivity.png`
- `cdp_worker_sensitivity.png`

## 4. 核心结果

Batch 敏感性：

| Batch | Worker 上限 | CDP 可行率 | CDP 平均时延/ms | CDP 平均卫星能耗/J | Oracle 选择 CDP 比例 |
|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 100.0% | 524.77 | 18.89 | 100.0% |
| 32 | 4 | 100.0% | 1002.02 | 37.62 | 100.0% |
| 64 | 4 | 100.0% | 1995.62 | 77.68 | 100.0% |
| 128 | 4 | 0.0% | N/A | N/A | 0.0% |

Worker 数敏感性：

| Batch | Worker 上限 | CDP 可行率 | CDP 平均时延/ms | CDP 平均卫星能耗/J | 平均实际 worker 数 |
|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 100.0% | 3888.96 | 77.35 | 2.00 |
| 64 | 3 | 100.0% | 2666.07 | 77.40 | 3.00 |
| 64 | 4 | 100.0% | 1995.62 | 77.68 | 4.00 |

## 5. 论文可写结论

1. 在 batch=16/32/64 且 worker 内存可满足完整模型部署时，CDP 在所有 STK 时间片中均可行，且 Oracle-Min-Latency 全部选择 CDP，说明数据并行模式具有稳定低时延优势。
2. 当 batch 增大到 128 时，CDP 可行率下降为 0%，说明 CDP 的低时延优势存在资源边界，固定使用 CDP 会导致任务失败。
3. 当 batch=64 时，worker 上限从 2 增加到 4，CDP 平均时延从 3888.96 ms 降至 1995.62 ms，说明 CDP 对可用 worker 数量敏感。
4. worker 数增加时，平均卫星能耗略有上升，说明 CDP 的低时延收益来自多星并行，代价是更高的多节点资源占用。
5. 这组实验可以支撑 FWMS 的稳定性叙事：模式选择不是单纯追求 CDP，而是在 CDP 可行时利用其低时延优势，在 CDP 不可行时切换到 PMP/GS-Only 等保底模式。
