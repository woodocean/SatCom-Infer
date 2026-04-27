# STK 动态 PMP 实验说明

本文档说明 `stk_dynamic_experiment.py` 的用途和基本用法。

## 实验口径

当前 STK 数据只用于两件事：

- 可见性约束：由 STK `Access Data` 判断链路是否可用。
- 传播时延：由 STK `AER Range` 计算 `delay_ms = range_km / 299792.458 * 1000`。

链路带宽不再由 Shannon 理论上界计算，而采用系统有效带宽设定值：

- 星间链路 `RS-SAT / SAT-SAT`：在 `1000-20000 Mbps` 随机采样。
- 星地链路 `SAT-GS`：在 `50-300 Mbps` 随机采样。

随机采样由 `--seed` 固定，因此同一命令可复现。

## 动态时间片流程

脚本会自动执行：

- 读取六个 STK 报告。
- 将场景时间切成固定时间片，默认 `5 min`。
- 每个时间片内搜索 `RS -> SAT -> ... -> GS` 候选路径。
- 默认最大跳数 `max_hops=6`。
- 选择共同可见持续时间 `common_duration_s` 最长的路径。
- 为该时间片生成独立 `network_config`。
- 调用 `Scheduler` 跑 PMP 六个算法。
- 写出时间片路径表和理论结果表。

## 常用命令

当前脚本默认从项目目录内读取：

```text
data/stk/
  Chain2_Access_Data.txt
  Chain2_Access_AER.txt
  Chain4_Access_Data.txt
  Chain4_Access_AER.txt
  Chain5_Access_Data.txt
  Chain5_Access_AER.txt
```

长期整理时，也可以按场景单独建目录：

```text
data/stk/sat_inferv2_500km_86p4deg_18x8_f2_20260414/
  Chain2_Access_Data.txt
  Chain2_Access_AER.txt
  Chain4_Access_Data.txt
  Chain4_Access_AER.txt
  Chain5_Access_Data.txt
  Chain5_Access_AER.txt
  scenario_meta.json
```

如果文件已经放在 `data/stk`，直接运行：

```powershell
python stk_dynamic_experiment.py `
  --time-start "14 Apr 2026 04:00:00.000" `
  --time-stop "14 Apr 2026 08:00:00.000" `
  --slot-minutes 5 `
  --max-hops 6 `
  --model-name yolov5 `
  --batch-size 32 `
  --input-h 640 `
  --input-w 640 `
  --run-id stk_dynamic_yolo_001
```

如果报告放在别的项目内目录，可以指定：

```powershell
--stk-dir data\stk\your_scenario_name
```

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--stk-dir` | `data/stk` | STK 报告目录 |
| `--slot-minutes` | `5` | 时间片长度 |
| `--max-hops` | `6` | RS 到 GS 的最大跳数 |
| `--isl-range-mbps` | `1000,20000` | 星间有效带宽随机范围 |
| `--gsl-range-mbps` | `50,300` | 星地有效带宽随机范围 |
| `--seed` | `42` | 控制带宽采样和随机算法 |
| `--repeat-per-slot` | `1` | 每个时间片重复运行次数 |
| `--output-csv` | `<output-dir>/results_long_stk_dynamic.csv` | 标准化结果表路径 |
| `--max-neighbors-per-node` | `24` | 路径搜索展开邻居数上限 |
| `--beam-width-per-node` | `8` | 每跳每个节点保留的 beam 宽度 |

如果搜索太慢，可以降低：

```powershell
--max-neighbors-per-node 12 --beam-width-per-node 4
```

如果候选路径太少，可以提高：

```powershell
--max-neighbors-per-node 32 --beam-width-per-node 12
```

## 输出文件

默认输出目录：

```text
result/stk_dynamic/<run_id>/
```

主要文件：

| 文件 | 说明 |
|---|---|
| `metadata.json` | 本次实验参数 |
| `stk_dynamic_slots.csv` | 每个时间片选中的路径、跳数、传播时延、平均带宽 |
| `results_long_stk_dynamic.csv` | 六个算法的理论时延和卫星能耗结果 |
| `configs/*.json` | 每个时间片独立生成的 `network_config` |
| `candidates/*.json` | 每个时间片的候选路径记录 |

## 如何验证结果

重点检查：

- `stk_dynamic_slots.csv` 中是否存在 `status=completed` 的时间片。
- `selected_path` 是否符合 `RS -> LEO... -> Shenzhen`。
- `pipeline_path` 是否映射为 `RS -> SAT-01 -> ... -> GS`。
- `total_propagation_delay_ms` 是否与 STK 距离量级一致。
- `isl_avg_bw_mbps` 是否在 `1000-20000` 范围内。
- `gsl_avg_bw_mbps` 是否在 `50-300` 范围内。

可以用其中一个时间片配置单独验证：

```powershell
python experiments_runner.py --config result\stk_dynamic\<run_id>\configs\slot_001_040500_041000_network_config.json --exp-mode theory --exp-type algo_effectiveness --num-tasks 5 --run-id stk_slot_check
```

如果希望直接追加到根目录长表，可以加：

```powershell
--output-csv results_long.csv
```

## 当前边界

- 不覆盖 `config/network_config.json`。
- 不启动 Jetson，也不做 SSH / Docker 管理。
- 不是实物仿真，只是 STK 驱动的理论拓扑实验。
- STK 不计算带宽，只提供可见性和传播时延。
- CDP 和模式选择暂未接入该脚本。
