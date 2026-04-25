# 实验结果归档说明

新的实验结果统一放在：

```text
result/runs/
```

旧的 `result/v1.0`、`result/v2.0`、`result/v3.0`、`result/v4.0` 可以继续保留作为历史版本；后续日常实验建议都走 `result/runs/`。

## 1. 每次实验的目录结构

每次运行 `experiments_runner.py` 会自动创建一个可追溯目录，例如：

```text
result/runs/20260425_183012__gsl_yolov5_b32_p20_r10__gsl_bw_yolov5_b32_640x640_theory_20to200_p20_r10_seed42/
```

目录内部结构：

```text
README.md
metadata.json
data/
  results_long_*.csv
  summary_*.csv
figures/
  *.png
config/
  network_config_snapshot.json
  experiment_presets_snapshot.json
```

## 2. 文件分别有什么用

`metadata.json`：

记录完整实验参数，适合机器读取。

`README.md`：

记录本次实验的可读摘要，适合人工快速查看。

`data/results_long_*.csv`：

从全局 `results_long.csv` 里切出来的本次实验数据。

`data/summary_*.csv`：

绘图脚本生成的聚合结果，例如每个带宽频点、每个算法的平均归一化时延、标准差和样本数。

`figures/*.png`：

本次实验对应的图片，不会再覆盖其他实验图片。

`config/*_snapshot.json`：

实验开始时的配置快照，方便以后复现实验。

## 3. 总目录

所有实验会追加记录到：

```text
result/EXPERIMENT_INDEX.md
```

这个文件相当于实验账本。以后找实验时，先看这里，再进入对应 run 目录。

## 4. 推荐命名方式

正式实验建议手动指定 `run_id`：

```bash
python experiments_runner.py --preset gsl_yolov5 --run-id gsl_yolov5_b32_20to200_p20_r10_seed42
```

`run_id` 推荐包含：

- 实验类型：`isl` 或 `gsl`
- 模型：`yolov5`、`resnet101`
- batch：例如 `b32`
- 扫参范围：例如 `20to200`
- 频点数：例如 `p20`
- 随机重复次数：例如 `r10`
- seed：例如 `seed42`

## 5. 推荐工作流

跑实验：

```bash
python experiments_runner.py --preset gsl_yolov5 --run-id gsl_yolov5_b32_20to200_p20_r10_seed42
```

画图并生成 summary：

```bash
python plot_avg_tho_vs_real.py --run-id gsl_yolov5_b32_20to200_p20_r10_seed42 --exp-type gsl_bandwidth_sensitivity --no-show
```

查看结果：

```text
result/EXPERIMENT_INDEX.md
result/runs/<本次实验目录>/
```

## 6. 多模型结果是否应该平均

不建议默认把不同模型直接平均。

不同模型的计算量、通信量、层结构差异很大，直接平均会让结果解释变模糊。更推荐：

- 每个模型、每个 batch 单独跑一组图。
- 如果论文需要总体趋势，再额外定义一个明确的 workload set。
- 对 workload set 求平均时，要在图名和说明中明确写出包含哪些模型和 batch。

