# experiments_runner.py Quickstart

这份文件是速查卡，只保留最常用命令。完整参数解释看 `doc/experiments_runner_usage.md`。

## 最常用命令

算法有效性理论实验：

```bash
python experiments_runner.py --preset algo
```

ISL 带宽敏感性实验：

```bash
python experiments_runner.py --preset isl
```

默认会在 500 到 20000 Mbps 之间生成 20 个等间隔点。确定性算法每个点跑 1 次，`Random/GA` 每个点重复 10 次取平均。

GSL 带宽敏感性实验：

```bash
python experiments_runner.py --preset gsl
```

默认会在 20 到 200 Mbps 之间生成 20 个等间隔点。确定性算法每个点跑 1 次，`Random/GA` 每个点重复 10 次取平均。

节点数敏感性实验：

```bash
python experiments_runner.py --preset nodes
```

默认会生成 1 到 5 个协作卫星的线性流水线拓扑。每个拓扑会生成 10 个随机资源场景，所有算法都在这些场景上取平均。

画最新一批结果：

```bash
python plot_avg_tho_vs_real.py --exp-type auto --no-show
```

实验数据、配置快照、图片和 summary 会归档到：

```text
result/runs/
```

总目录：

```text
result/EXPERIMENT_INDEX.md
```

## 常用自定义

指定批次名：

```bash
python experiments_runner.py --preset isl --run-id test_isl_001
```

指定随机种子：

```bash
python experiments_runner.py --preset algo --seed 123
```

用范围和点数自动生成带宽扫频点：

```bash
python experiments_runner.py --preset isl --sweep-start 500 --sweep-stop 20000 --sweep-points 30 --repeat-per-point 10
```

使用配置文件里的自定义 preset：

```bash
python experiments_runner.py --preset isl_yolo
```

使用 YOLOv5 的节点数敏感性 preset：

```bash
python experiments_runner.py --preset nodes_yolo
```

其他模型的节点数敏感性 preset：

```bash
python experiments_runner.py --preset nodes_resnet101
python experiments_runner.py --preset nodes_vgg19
python experiments_runner.py --preset nodes_swin_base
python experiments_runner.py --preset nodes_vit_huge
```

默认自定义 preset 文件是：

```text
config/experiment_presets.json
```

## 两个实验文档怎么分工

`doc/experiments_runner_quickstart.md`：

只放高频命令，适合跑实验前快速复制。

`doc/experiments_runner_usage.md`：

放完整说明，包括参数含义、preset 怎么写、结果表怎么理解、随机性和带宽扫参口径。

`doc/experiment_result_archive.md`：

说明实验结果、图片、summary 和总目录如何归档。
