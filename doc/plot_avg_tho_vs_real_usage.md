# plot_avg_tho_vs_real.py 使用说明

`plot_avg_tho_vs_real.py` 从统一长表 `results_long.csv` 读取实验结果并自动画图。

它会根据 `exp_type` 自动选择图形：

- `algo_effectiveness`：算法对比柱状图
- `isl_bandwidth_sensitivity`：ISL 带宽敏感性折线图
- `gsl_bandwidth_sensitivity`：GSL 带宽敏感性折线图
- `node_count_sensitivity`：协作节点数敏感性折线图

## 1. 最常用命令

自动识别最新一批实验并画图：

```bash
python plot_avg_tho_vs_real.py --exp-type auto
```

只保存图片，不弹出窗口：

```bash
python plot_avg_tho_vs_real.py --exp-type auto --no-show
```

## 2. 指定实验类型

```bash
python plot_avg_tho_vs_real.py --exp-type algo_effectiveness
python plot_avg_tho_vs_real.py --exp-type isl_bandwidth_sensitivity
python plot_avg_tho_vs_real.py --exp-type gsl_bandwidth_sensitivity
python plot_avg_tho_vs_real.py --exp-type node_count_sensitivity
```

## 3. 指定 run_id

如果不传 `--run-id`，脚本默认读取最新一批 `run_id`。

指定批次：

```bash
python plot_avg_tho_vs_real.py --run-id paper_isl_001 --exp-type isl_bandwidth_sensitivity
```

## 4. 输入和输出

读取指定 CSV：

```bash
python plot_avg_tho_vs_real.py --results-csv results_long.csv --exp-type auto
```

指定输出图片路径：

```bash
python plot_avg_tho_vs_real.py --exp-type gsl_bandwidth_sensitivity --output gsl_line.png --no-show
```

如果不指定 `--output`，图片会自动保存到：

```text
result/runs/<本次实验目录>/figures/
```

同时会生成聚合 summary：

```text
result/runs/<本次实验目录>/data/summary_*.csv
```

## 5. 参数说明

- `--results-csv`：输入结果长表，默认 `results_long.csv`
- `--run-id`：指定实验批次，不传则自动取最新批次
- `--exp-type`：绘图类型，默认 `auto`
- `--output`：输出图片路径
- `--no-show`：只保存图片，不调用 `plt.show()`

## 6. 图中指标

纵轴默认使用：

```text
norm_latency_vs_gs
```

含义是相对 `GS-Only` 的归一化时延。

- `1.0` 表示和 `GS-Only` 一样
- 小于 `1.0` 表示比 `GS-Only` 更快
- 大于 `1.0` 表示比 `GS-Only` 更慢

带宽敏感性图的横轴：

- ISL 图使用 `isl_avg_bw_mbps`
- GSL 图使用 `gsl_avg_bw_mbps`

节点数敏感性图的横轴：

- 优先使用 `pipeline_node_count`
- 旧数据没有该列时，退回使用 `sweep_value`

如果同一个频点有多次重复测量，脚本会按：

```text
带宽频点 + algorithm
```

自动取 `norm_latency_vs_gs` 的平均值后再画线。

带宽敏感性实验里，通常只有 `Random` 和 `GA` 会有多次重复测量；确定性算法每个频点只写入一次。

节点数敏感性实验里，所有算法都会在多个随机资源场景上重复测量，脚本按节点数和算法自动取平均。

## 7. 推荐工作流

跑实验：

```bash
python experiments_runner.py --preset isl --run-id paper_isl_001
```

画图：

```bash
python plot_avg_tho_vs_real.py --run-id paper_isl_001 --exp-type isl_bandwidth_sensitivity --no-show
```

查看归档：

```text
result/EXPERIMENT_INDEX.md
result/runs/<本次实验目录>/
```
