# experiments_runner.py 完整说明

`experiments_runner.py` 是实验编排入口。它负责生成实验任务、更新网络配置、调用调度器，并把理论结果写入统一长表 `results_long.csv`。

`doc/experiments_runner_quickstart.md` 是速查卡；本文件是完整说明。

## 1. 当前能做什么

目前支持三个功能模块：

- `algo_effectiveness`：算法有效性实验，用随机网络条件比较不同调度算法。
- `isl_bandwidth_sensitivity`：星间链路带宽敏感性实验，只扫 ISL 平均带宽。
- `gsl_bandwidth_sensitivity`：星地链路带宽敏感性实验，只扫 GSL 平均带宽。

带宽敏感性实验当前是理论模式。它固定模型、batch、输入尺寸，只改变目标链路类别的平均带宽。

## 2. 推荐用法

最推荐先用 preset：

```bash
python experiments_runner.py --preset algo
python experiments_runner.py --preset isl
python experiments_runner.py --preset gsl
```

这三个是内置 preset。你也可以在 `config/experiment_presets.json` 里添加自己的 preset。

## 3. 参数分层

高频参数：

- `--preset`：快捷实验配置，例如 `algo`、`isl`、`gsl`、`isl_yolo`
- `--run-id`：实验批次名，不传会自动生成
- `--seed`：基础随机种子，默认 `42`

实验类型参数：

- `--exp-type`：实验类型，可选 `algo_effectiveness`、`isl_bandwidth_sensitivity`、`gsl_bandwidth_sensitivity`
- `--exp-mode`：实验模式，可选 `theory`、`physical`、`hybrid`
- `--num-tasks`：算法有效性实验的任务数量

带宽敏感性参数：

- `--sweep-values`：扫参点，逗号分隔，例如 `500,1000,2000`
- `--sweep-start`：扫参范围起点，单位 Mbps
- `--sweep-stop`：扫参范围终点，单位 Mbps
- `--sweep-points`：扫参点数，脚本会在起点和终点之间均匀生成
- `--fixed-model`：固定模型，默认 `yolov5`
- `--fixed-batch-size`：固定 batch，默认 `32`
- `--fixed-input-h`：固定输入高度，默认 `640`
- `--fixed-input-w`：固定输入宽度，默认 `640`
- `--repeat-per-point`：每个带宽频点重复测量次数，默认 `10`

底层配置参数：

- `--config`：网络配置文件，默认 `config/network_config.json`
- `--rs-id`：RS 节点 ID，默认 `RS`
- `--preset-file`：preset JSON 文件，默认 `config/experiment_presets.json`

## 4. 如何设置自己的 preset

编辑 `config/experiment_presets.json`，添加一个新键即可。

例子：

```json
{
  "my_isl_resnet": {
    "exp_type": "isl_bandwidth_sensitivity",
    "exp_mode": "theory",
    "sweep_values": "500,1000,2000,5000",
    "fixed_model": "resnet101",
    "fixed_batch_size": 32,
    "fixed_input_h": 224,
    "fixed_input_w": 224
  }
}
```

运行：

```bash
python experiments_runner.py --preset my_isl_resnet
```

preset 里的字段名要和命令行参数去掉 `--` 后一致，例如 `--fixed-model` 写成 `fixed_model`。

如果不想手写每个频点，可以用范围写法：

```json
{
  "my_isl_range": {
    "exp_type": "isl_bandwidth_sensitivity",
    "exp_mode": "theory",
    "sweep_start": 500,
    "sweep_stop": 20000,
    "sweep_points": 30,
    "repeat_per_point": 10,
    "fixed_model": "yolov5",
    "fixed_batch_size": 32,
    "fixed_input_h": 640,
    "fixed_input_w": 640
  }
}
```

也可以直接在命令行写：

```bash
python experiments_runner.py --preset isl --sweep-start 500 --sweep-stop 20000 --sweep-points 30 --repeat-per-point 10
```

`--sweep-values` 优先级更高；如果同时给了 `--sweep-values` 和范围参数，脚本会使用 `--sweep-values`。

默认内置 preset 使用 20 个等间隔点：

- `isl`：500 到 20000 Mbps，共 20 个点
- `gsl`：20 到 200 Mbps，共 20 个点
- 每个频点默认对 `Random/GA` 重复 10 次，使用不同随机种子；确定性算法每个频点只运行 1 次

## 5. 带宽敏感性实验口径

`isl_bandwidth_sensitivity` 的 sweep 值表示 ISL 平均带宽。

`gsl_bandwidth_sensitivity` 的 sweep 值表示 GSL 平均带宽。

脚本不会把每条链路都改成完全一样的带宽，而是按基线比例缩放同类链路。这样可以保留链路之间的异构性，同时让这一类链路的平均带宽等于目标扫参值。

## 6. 随机性说明

脚本会根据 `seed + run_id + exp_type + task_id` 固定随机种子。

这会影响：

- 算法有效性实验中的随机网络状态。
- `Random` 算法的随机切分。
- `GA` 算法的随机初始化、交叉和变异。

带宽敏感性实验会对每个频点分两段运行：

- `LA-DP`、`Greedy`、`Uniform`、`GS-Only` 是确定性算法，每个频点只运行 1 次。
- `Random` 和 `GA` 会按 `repeat_per_point` 重复运行，每次使用不同 seed，绘图时按带宽和算法取平均。

如果你想复现实验，建议显式指定 `--run-id` 和 `--seed`。

例子：

```bash
python experiments_runner.py --preset isl --run-id paper_isl_001 --seed 42
```

## 7. 结果文件

理论结果会写入：

```text
results_long.csv
```

同时兼容旧流程写入：

```text
theoretical_results.csv
```

`results_long.csv` 是统一长表。关键字段：

- `run_id`
- `exp_type`
- `mode`
- `task_id`
- `algorithm`
- `model_name`
- `batch_size`
- `input_h`
- `input_w`
- `isl_avg_bw_mbps`
- `gsl_avg_bw_mbps`
- `latency_ms`
- `norm_latency_vs_gs`
- `timestamp`

长表不是混乱，而是方便筛选和分组。比如：

```python
df = pd.read_csv("results_long.csv")
isl_df = df[df["exp_type"] == "isl_bandwidth_sensitivity"]
summary = isl_df.groupby(["isl_avg_bw_mbps", "algorithm"])["norm_latency_vs_gs"].mean()
```

## 8. 必要提醒

`experiments_runner.py` 会写回 `config/network_config.json`。理论实验通常没问题，但如果你要保留某个手工网络配置，跑实验前最好先确认当前配置。

`physical` 和 `hybrid` 目前主要服务 `algo_effectiveness`。带宽敏感性实验当前会按理论实验执行。
