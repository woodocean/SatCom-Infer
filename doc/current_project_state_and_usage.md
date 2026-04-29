# 当前项目状态与使用说明

本文档是当前仓库的“项目地图”。它只整理代码入口、核心模块、实验产物格式、当前完成度和剩余实验，不替代论文正文。

更新时间：2026-04-28

## 1. 当前主线

现在项目可以分成三条主线：

| 主线 | 入口 | 作用 | 当前状态 |
|---|---|---|---|
| 传统 PMP 实验 | `experiments_runner.py` | 固定或随机拓扑下比较 LA-DP、Greedy、Uniform、Random、GA、GS-Only | 已能跑理论、能耗、带宽敏感性、节点数敏感性，少量实物/混合模式可用 |
| STK 动态 PMP 实验 | `stk_dynamic_experiment.py` | 把 STK 可见性切成时间片，每个时间片选一条路径并跑 PMP 系算法 | 已完成 5 模型 STK PMP 结果和跨模型汇总 |
| STK 模式选择实验 | `mode_selection_experiment.py` | 以 STK 时间片作为场景，统一评估 PMP、GS-Only、Sat-Only、CDP、FWMS | 已到 Stage 5，5 模型结果和跨模型图已生成 |

不要把这三条混在一个口径里讲。传统 PMP 实验回答“同一路径上哪个 PMP 算法好”；模式选择实验回答“同一个 STK 时间片下哪个推理模式好”。

## 2. 入口脚本职责

| 文件 | 负责什么 | 不负责什么 |
|---|---|---|
| `main.py` | 启动单个节点进程，读取 `config/network_config.json`，注册邻居，等待任务 | 不编排实验，不生成理论结果 |
| `experiments_runner.py` | 传统实验编排，更新拓扑，调用 `Scheduler`，写 `results_long.csv` 和归档 | 不处理 STK 时间片，不做模式选择 |
| `stk_dynamic_experiment.py` | 读取 STK 报告，切时间片，搜索候选路径，选择 `selected_path`，生成每片配置并跑 PMP | 不比较 CDP / Sat-Only / FWMS |
| `mode_selection_experiment.py` | 从已有 STK run 读取 slot scene，评估五类模式，输出模式选择结果 | 不启动 Jetson，不做实物测量 |
| `build_stk_network_config.py` | 从 STK 报告导出候选路径和若干静态 `network_config` | 不跑调度算法 |
| `plot_avg_tho_vs_real.py` | 传统长表和 STK PMP 图的通用绘图入口 | 不画 Stage5 模式选择跨模型图 |
| `plot_stk_cross_model_summary.py` | 汇总多个 STK PMP run，输出跨模型 PMP 表和图 | 不处理模式选择 Stage5 |
| `plot_mode_selection_summary.py` | 汇总多个 Stage5 模式选择 run，输出跨模型模式选择表和图 | 不跑实验，只读已有结果 |

## 3. 核心模块职责

| 模块 | 核心职责 | 当前实现要点 |
|---|---|---|
| `core/scheduler.py` | 把任务 profile 和网络配置转成求解器输入，调用 PMP 算法，写标准化长表 | 输出字段统一到 `STANDARDIZED_RESULT_FIELDS`，理论能耗作为评价指标 |
| `algorithms/pmp_solver.py` | PMP / LADP / Greedy / Uniform / Random / GA / GS-Only 求解 | `solve_la_dp` 是主算法，`solve_bent_pipe` 是 GS-Only，内存检查已经读取 `hardware.memory_mb` |
| `algorithms/cdp_solver.py` | 无聚合器版 CDP / LAWA 数据分配 | 建模为 RS 分发到多 worker，worker 全模型推理，各自直接回 GS |
| `core/mode_scene_builder.py` | 把 STK 动态实验结果整理成统一 `SlotScene` | 每个 completed slot 变成一个场景对象 |
| `core/mode_evaluators.py` | 模式选择中的 PMP、GS-Only、Sat-Only、CDP 评估器 | 每个模式使用自己的候选结构和路径策略 |
| `core/stk_parser.py` | 解析 STK Access Data 和 AER 报告 | 输出可见窗口、距离采样、传播时延 |
| `core/stk_scenario_builder.py` | 基于 STK 可见性构建候选路径和网络配置 | Chain2 是 SAT-GS，Chain4 是 SAT-SAT，Chain5 是 RS-SAT |
| `core/node.py` | 实物节点运行逻辑 | PMP 实物链路可用；CDP 实物逻辑仍有早期聚合器字段，暂不等同于论文无聚合器 CDP |
| `core/inference.py` | 真实模型加载和分层推理 | 支持 YOLOv5、ResNet、VGG19、Swin-Base、ViT-Huge 等 DAG wrapper |
| `core/communicate.py` | 节点间 UDP 分包通信 | 实物传输测量和网络收发基础 |

`algorithms/mode_selector.py` 当前基本是空占位。真正的 FWMS 第一版逻辑现在在 `mode_selection_experiment.py` 的 `_select_fwms` 中。

## 4. 当前模式定义

| 模式 | 定义 | 当前实现位置 |
|---|---|---|
| PMP | 沿 STK 动态实验已经选出的 `selected_path` 运行 LA-DP 分层流水线 | `evaluate_pmp_slot` |
| GS-Only | 为 GS-Only 在候选路径里选预测时延最低的路由，所有层在 GS 运行 | `evaluate_gs_only_slot` |
| Sat-Only | 枚举候选路径里的单颗卫星，选择能放下全模型且总时延最低的单星执行方案 | `evaluate_sat_only_slot` |
| CDP | RS 分发 batch 到多颗 worker，worker 各自全模型推理并直接回 GS，无星上聚合器 | `evaluate_cdp_slot` + `CDPSolver.solve_lawa_discrete` |
| FWMS | 过滤不可行模式，在可行模式中选择预测端到端时延最低者，同等时延下用卫星能耗打破平局 | `_select_fwms` |

需要注意：当前 FWMS 还不是复杂学习型选择器，它是 prediction-based 模式选择。它的价值在于把可行性、路径、内存、STK 可见性和各模式预测结果统一进同一个决策表。

## 5. 当前模型库

当前 profile 数据库中保留 5 个真实模型：

| 模型键名 | 论文/图中建议名称 | 典型用途 |
|---|---|---|
| `yolov5` | YOLOv5 | 大输入图像检测任务 |
| `resnet101` | ResNet101 | CNN 分类基线 |
| `vgg19` | VGG19 | 参数量较大、内存压力明显的链式 CNN |
| `swin_base` | Swin-Base | Transformer/CV backbone |
| `vit_huge` | ViT-Huge | 大参数视觉 Transformer，单星/CDP 容易受内存约束 |

`convnext_xxl` 已按要求移除，不再作为论文实验模型。

## 6. 结果目录结构

| 目录 | 内容 |
|---|---|
| `result/runs/` | 传统 `experiments_runner.py` 归档结果 |
| `result/stk_dynamic/<run_id>/` | STK 动态 PMP 实验结果 |
| `result/stk_dynamic/cross_model/` | STK PMP 跨模型总表和图 |
| `result/mode_selection/<run_id>/` | 单个模型的模式选择 Stage 结果 |
| `result/mode_selection/cross_model_stage5/` | Stage5 模式选择跨模型总表和图 |

本次只抽样查看了 CSV/JSON 的文件名、表头和开头内容，没有逐条检查全部实验记录。

## 7. 关键 CSV / JSON 格式

### `results_long.csv`

传统实验和 STK PMP 结果长表，表头如下：

```text
run_id,exp_type,mode,task_id,algorithm,model_name,batch_size,input_h,input_w,isl_avg_bw_mbps,gsl_avg_bw_mbps,pipeline_node_count,pipeline_hop_count,pipeline_path,sweep_param,sweep_value,latency_ms,norm_latency_vs_gs,energy_compute_j,energy_comm_j,energy_total_j,satellite_energy_j,norm_energy_vs_gs,satellite_compute_time_ms,satellite_tx_time_ms,energy_model,timestamp
```

### `result/stk_dynamic/<run_id>/stk_dynamic_slots.csv`

每个 STK 时间片的路径选择表，表头如下：

```text
run_id,slot_id,slot_start,slot_stop,status,selected_path,pipeline_path,hop_count,satellite_count,common_start,common_stop,common_duration_s,total_range_km,total_propagation_delay_ms,isl_avg_bw_mbps,gsl_avg_bw_mbps,config_path,note
```

### `result/mode_selection/<run_id>/data/slot_mode_results.csv`

模式选择核心结果表，表头如下：

```text
run_id,source_run_id,slot_id,mode_family,mode_algo,candidate_id,route_policy,feasible,reason,latency_ms,satellite_energy_j,energy_compute_j,energy_comm_j,satellite_compute_time_ms,satellite_tx_time_ms,active_sat_count,hop_count,route,pipeline_path,plan_json,config_path,candidate_path,timestamp
```

### `summary_by_mode.csv`

每种模式的平均指标：

```text
mode_family,rows,feasible_rows,feasible_rate,avg_latency_ms,avg_satellite_energy_j
```

### `fwms_selection_distribution.csv`

FWMS 在一个模型的所有时间片上选择了哪些模式：

```text
selected_mode,count,ratio
```

### `metadata.json`

记录 run id、实验类型、阶段、来源 STK run、有效任务规格、已实现模式、路径策略、输出路径等。

## 8. 常用命令

### 传统 PMP 理论实验

```powershell
python experiments_runner.py --preset algo
python experiments_runner.py --preset isl_yolo
python experiments_runner.py --preset gsl_yolo
python experiments_runner.py --preset nodes_yolo
python experiments_runner.py --preset energy_yolo
```

### STK 动态 PMP

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

### 单模型模式选择 Stage5

```powershell
python mode_selection_experiment.py `
  --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 `
  --run-id mode_selection_yolo_stage5_fwms_b64 `
  --batch-size-override 64
```

### 跨模型模式选择绘图

```powershell
python plot_mode_selection_summary.py
```

输出目录：

```text
result/mode_selection/cross_model_stage5/
```

## 9. 当前已完成的模式选择结果

从 `result/mode_selection/cross_model_stage5/mode_selection_cross_model_summary.csv` 抽样汇总：

| 模型 | FWMS 主要选择 | 重要现象 |
|---|---|---|
| YOLOv5 | CDP 42/42 | CDP 全部可行且平均时延最低 |
| ResNet101 | CDP 42/42 | CDP 明显低于 PMP 和 GS-Only |
| Swin-Base | CDP 42/42 | CDP 明显低于 PMP 和 GS-Only |
| VGG19 | GS-Only 29/42，PMP 13/42 | CDP 和 Sat-Only 因内存不可行，FWMS 在 GS-Only 与 PMP 间切换 |
| ViT-Huge | GS-Only 42/42 | CDP 和 Sat-Only 因内存不可行，PMP 可行但平均时延高于 GS-Only |

当前跨模型输出文件：

```text
mode_selection_cross_model_summary.csv
fwms_selection_cross_model.csv
mode_selection_avg_latency.png
mode_selection_avg_energy.png
mode_selection_feasible_rate.png
fwms_selection_distribution.png
```

## 10. 还差哪些实验

优先级从高到低：

| 优先级 | 实验 | 目的 | 当前建议 |
|---|---|---|---|
| 高 | Jetson 半实物补量 | 回应老师和评阅意见，证明理论趋势能落到设备上 | 2 模型、2 batch、3 模式、3 次重复，先拿最小矩阵 |
| 高 | CDP batch sensitivity | 解释为什么 CDP 需要较大 batch 才稳定发挥 | 扫 `8/16/32/64` |
| 中 | CDP worker count sensitivity | 解释 worker 数量收益和边界 | 扫 `2/3/4` |
| 中 | 图表统一美术风格 | 让论文图可读且风格一致 | 当前 Stage5 图已统一一版，但旧图还没洗 |
| 中 | 实验章节结论整理 | 每张图配一句可写进论文的结论 | 与 PPT 同步做 |
| 低 | 复杂能耗预算约束 | 更完整的资源约束模式选择 | 可放未来工作 |
| 低 | 混合 pipeline-data parallelism | 解决 CDP 内存不可行但仍想并行的问题 | 本科阶段不建议展开 |

## 11. PMP 为何可能比 GS-Only 慢

这次 Stage5 中，除 YOLOv5 外，PMP 平均时延常常高于 GS-Only，主要不是 LA-DP 算法突然失效，而是实验口径变了。

关键原因有三个：

| 原因 | 解释 |
|---|---|
| 路由口径不同 | PMP 固定使用 `stk_dynamic_experiment.py` 选出的 `selected_path`，该路径按共同可见时间优先，不一定是最低时延路径；GS-Only 在模式选择里会重新枚举候选路径并选预测时延最低路径 |
| 通信/计算权衡不同 | ResNet、VGG、Swin、ViT 的输入是 `224x224`，原始输入通信量小，GS 计算很强，因此把数据直接送到 GS 可能比在卫星上计算再传中间特征更快 |
| PMP 中间特征未必压缩 | 对某些模型，早期或中间层特征没有明显小于原始输入，甚至可能膨胀，PMP 的分层通信收益不足 |

所以你记得“PMP 算法对比实验里 LADP 比 GS-Only 好”并不矛盾。旧实验是在同一路径、同一拓扑、同一随机资源场景下比较 PMP 系算法；现在 Stage5 是模式选择实验，GS-Only 可以用自己的最优路径，而且模型输入尺寸和内存约束也不同。

如果要让论文表述严谨，可以写成：

```text
在传统 PMP 算法对比实验中，LA-DP 在给定流水线路径上优于其他切分策略及同路径 GS-Only 基线。
在 STK 动态模式选择实验中，不同模式允许使用符合其定义的执行结构，因此 GS-Only 可选择更适合回传地面站的路径。
对于输入较小且中间特征压缩不明显的模型，GS-Only 可能优于固定路径 PMP。
这恰好说明模式选择有必要：不同模型和场景下，不应固定采用单一推理模式。
```

## 12. 当前最容易混淆的边界

- STK 提供可见性和传播时延，不直接提供有效带宽。
- 当前有效带宽是按 ISL/GSL 范围随机采样并用 seed 固定。
- `stk_dynamic_experiment.py` 的 `selected_path` 当前按 `max_common_duration_s` 选择，不是最小时延路径。
- PMP Stage5 当前不枚举所有 PMP 路径，只沿 `selected_path` 运行，这符合“路由已由上层给定”的论文假设。
- GS-Only / Sat-Only / CDP 在 Stage5 中会使用自己的候选结构，因此不能直接和旧 PMP 算法对比图混成一个结论。
- 理论 CDP 已经是无聚合器版；实物侧 `core/node.py` 还保留早期 CDP 聚合器消息路径，后续如果做 CDP 实物，需要单独对齐实现口径。
- 当前能耗是理论卫星侧能耗指标，不是 Jetson 实测功耗。

