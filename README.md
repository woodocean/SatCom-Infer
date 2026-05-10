# SatCom-Infer

面向分布式卫星协同推理的实验项目。  
项目关注这样一个问题：当遥感卫星、LEO 计算卫星和地面站共同参与深度神经网络推理时，如何在**流水线模型切分**与**数据并行**两类协作模式之间做出合理选择，并在通信、算力、内存、可见性等约束下尽量降低端到端时延。

## 项目在做什么

当前仓库主要围绕三条实验主线展开：

1. `PMP` 模式下的模型切分与算法对比  
   比较 `LADP / Greedy / GA / Random / Uniform / GS-Only` 等算法在流水线协同推理中的表现。

2. `STK` 动态拓扑下的理论实验  
   将 STK 可见性报告切分为多个时间片，在动态链路条件下评估协同推理策略。

3. `FWMS` 模式选择实验  
   在 `PMP / CDP / GS-Only / Sat-Only` 之间进行统一评估，研究不同任务和资源条件下的模式适用边界。

## 项目意义

相比只研究单一推理模式，这个项目更关注：

- 不同协同模式在星地环境中的适用边界
- 通信压缩收益、内存压力、链路瓶颈对推理模式的影响
- 动态拓扑和异构资源条件下的稳定性与可行性

因此它更像一个**协同推理实验平台 + 论文实验仓库**，而不是单一算法 demo。

## 核心能力

- 支持 `PMP` 流水线模型切分推理
- 支持 `CDP` 数据并行推理评估
- 支持 `FWMS` 模式选择实验
- 支持基于 STK 可见性报告生成动态网络场景
- 支持从已有结果重绘论文图
- 保留半实物验证入口，便于后续接入 PC + Jetson 平台

## 仓库结构

### 核心代码

- [`algorithms/`](./algorithms)
  - 算法实现，如 `pmp_solver.py`、`cdp_solver.py`、`mode_selector.py`
- [`core/`](./core)
  - 调度、节点逻辑、场景构建、STK 解析、模式评估等底层模块
- [`models/`](./models)
  - 模型与分层 wrapper
- [`config/`](./config)
  - 网络配置、profile、设备参数

### 实验入口

- [`experiments_runner.py`](./experiments_runner.py)
  - 传统 PMP 理论实验主入口
- [`stk_dynamic_experiment.py`](./stk_dynamic_experiment.py)
  - STK 动态拓扑 PMP 实验主入口
- [`mode_selection_experiment.py`](./mode_selection_experiment.py)
  - 模式选择实验主入口
- [`thesis_entry.py`](./thesis_entry.py)
  - 统一入口，便于调用实验、绘图和工具脚本

### 工具与结果

- [`tools/`](./tools)
  - 前处理、绘图、半实物工具脚本
- [`doc/`](./doc)
  - 使用说明、实验路线、项目状态说明
- [`result/`](./result)
  - 全部实验结果、论文图、归档产物

## 快速开始

### 1. 查看统一入口

```powershell
python thesis_entry.py --help
```

### 2. 常用命令

传统 PMP 理论实验：

```powershell
python thesis_entry.py run-legacy --preset algo
```

STK 动态拓扑 PMP 实验：

```powershell
python thesis_entry.py run-stk --help
```

模式选择实验：

```powershell
python thesis_entry.py run-mode --help
```

从已有结果重画论文图：

```powershell
python thesis_entry.py plot-paper
```

### 3. 论文图与验收入口

如果你是为了快速查看当前论文结果，优先看：

- [`result/paper_figures_v2/README.md`](./result/paper_figures_v2/README.md)

这里统一整理了：

- 当前最终图放在哪里
- 每个实验的参数口径
- 重跑命令
- 结题验收时建议怎么展示

## 推荐阅读顺序

如果你第一次接触这个仓库，建议按下面顺序了解：

1. 先看本 README，理解项目目标和主线
2. 再看 [`doc/current_project_state_and_usage.md`](./doc/current_project_state_and_usage.md)
3. 想看实验路线时，再看 [`doc/experiment_entrypoints_and_mode_selection_plan.md`](./doc/experiment_entrypoints_and_mode_selection_plan.md)
4. 想直接看最终实验图时，打开 [`result/paper_figures_v2/README.md`](./result/paper_figures_v2/README.md)

## 说明

仓库在开发过程中积累过较多中间脚本、旧图和历史结果。当前已经逐步收口为：

- 根目录保留主实验入口
- 工具脚本统一收进 `tools/`
- 最终论文图和验收口径统一收进 `result/paper_figures_v2/`

如果发现旧文档、旧图或旧结果与当前主线不完全一致，优先以：

- [`result/paper_figures_v2/README.md`](./result/paper_figures_v2/README.md)
- [`doc/acceptance_structure_and_commands.md`](./doc/acceptance_structure_and_commands.md)

为准。
