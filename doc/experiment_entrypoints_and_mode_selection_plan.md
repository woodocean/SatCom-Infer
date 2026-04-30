# 实验入口与模式选择路线图

本文档用于回答当前阶段最容易混淆的四个问题：

1. 第一版模式选择实验到底怎么跑。
2. PMP 是否必须遍历一个时间片内的所有候选路径。
3. 当前各实验入口分别负责什么，彼此是什么关系。
4. 后续接入实物验证时，是否需要新增实验入口。

> 最新完整口径请优先看 `doc/mode_selection_boundary_and_stability_summary.md`。本文档保留路线图思路，并同步到当前 Stage 6 状态：`PMP / GS-Only / Sat-Only / CDP / FWMS-Feature / Oracle-Min-Latency` 已全部进入模式选择主链。

## 1. 当前实验入口总览

### `main.py`

用途：启动单个节点进程。

- 它不是实验编排脚本。
- 它读取固定的 `config/network_config.json`。
- 它的职责是把某个节点启动起来，等待实验编排器下发任务。

适用场景：

- 手动启动 `RS` / `SAT-*` / `GS` 节点。
- 配合 `experiments_runner.py --exp-mode physical|hybrid` 做实物或半实物验证。

### `experiments_runner.py`

用途：当前主实验编排入口。

- 它会更新网络配置、调用 `Scheduler`，并把结果写入统一长表 `results_long.csv`。
- 它当前支持的 `exp_type` 主要是：
  - `algo_effectiveness`
  - `energy_comparison`
  - `isl_bandwidth_sensitivity`
  - `gsl_bandwidth_sensitivity`
  - `node_count_sensitivity`
- 其中只有 `algo_effectiveness` 真正支持 `physical/hybrid`；其他类型目前仍是理论实验。

适用场景：

- 单一拓扑下比较 PMP 系算法。
- 扫带宽、扫节点数、扫能耗等理论实验。

### `build_stk_network_config.py`

用途：从 STK 报告中提取候选路径，并导出静态 `network_config`。

- 它只负责“建图”和“导出候选路径”。
- 它本身不跑调度算法。

适用场景：

- 想看某个时间窗里有哪些 `RS -> ... -> GS` 候选路径。
- 想把 STK 场景导成若干条可直接送给现有 PMP 调度器的静态配置。

### `stk_dynamic_experiment.py`

用途：STK 驱动的动态 PMP 理论实验入口。

- 它会把 STK 场景切成多个时间片。
- 每个时间片先搜索候选路径，再选出一条 `selected_path`。
- 然后基于该路径生成该时间片的 `network_config`，并调用 `Scheduler` 运行 PMP 相关算法。
- 结果输出到 `result/stk_dynamic/<run_id>/`。

重要边界：

- 它是 STK 驱动的理论拓扑实验，不是实物实验。
- 它当前只覆盖 PMP 线性路径场景。
- `CDP` 和 `FWMS` 还没有正式接入这条脚本。

### `plot_avg_tho_vs_real.py`

用途：当前通用绘图入口。

- 读取 `results_long.csv` 或 `results_long_stk_dynamic.csv`。
- 根据 `exp_type` 自动选择不同画法。
- 当前也支持 `stk_dynamic_pmp`。

### `plot_stk_cross_model_summary.py`

用途：把多个 STK run 聚合成跨模型总表和总图。

- 它读取每个 STK run 的 `summary_csv`。
- 输出跨模型总表、对比图和稳定性分析结果。

适用场景：

- 已经跑完多个模型的 STK 动态 PMP 实验后，做论文级汇总。

## 2. 这些入口之间的关系

可以把它们理解成三层：

### 第 1 层：节点运行层

- `main.py`

这是“把节点跑起来”的层，不负责实验设计。

### 第 2 层：实验编排层

- `experiments_runner.py`
- `stk_dynamic_experiment.py`

这是“决定跑什么实验、写什么结果表”的层。

区别是：

- `experiments_runner.py` 面向传统单拓扑理论实验，以及少量 `algo_effectiveness` 的实物/混合模式。
- `stk_dynamic_experiment.py` 面向 STK 时间片动态场景，但当前仍然只服务 PMP。

### 第 3 层：分析与画图层

- `plot_avg_tho_vs_real.py`
- `plot_stk_cross_model_summary.py`

这是“读结果表、出 summary、生成论文图表”的层。

## 3. 第一版模式选择实验推荐怎么跑

第一版不要直接改写现有 `experiments_runner.py` 或 `stk_dynamic_experiment.py` 的主流程。

更稳的做法是新增一条独立入口，例如：

- `mode_selection_experiment.py`

它的职责是：

1. 读取一个时间片场景。
2. 分别评估 `PMP / GS-Only / Sat-Only / CDP`。
3. 生成统一的模式结果表。
4. 再运行 `FWMS` 在模式之间做选择。

推荐执行顺序：

1. 先做 `slot_scene`。
2. 先接 `PMP`。
3. 再接 `GS-Only`。
4. 再接 `Sat-Only`。
5. 最后接无聚合器版 `CDP`。
6. 四种模式统一出结果后，再接 `FWMS`。

## 4. PMP 是否必须遍历所有候选路径

不必须。

这里要区分两种口径：

### 口径 A：论文假设“路由已给定”

如果论文里假设 PMP 的路由已经由外部路由层选好，那么第一版模式选择实验完全可以沿用这个假设：

- 每个时间片只给 `PMP` 一个既定路由。
- 这个路由可以直接使用当前 `stk_dynamic_experiment.py` 选出的 `selected_path`。
- 然后只在这条路径上运行 `LA-DP` 等算法。

这种做法的优点是：

- 和论文假设一致。
- 不把“路由选择”和“模式选择”混在一起。
- 第一版更容易解释。

### 口径 B：系统扩展版“在候选路径中挑最好的一条”

这是一种工程增强版做法：

- 在一个时间片内枚举多条候选 `PMP` 路径。
- 对每条路径分别运行 `LA-DP`。
- 取其中最好的一条作为该时间片的 `PMP` 成绩。

这种做法不是错，但它会把：

- 路由选择能力
- PMP 调度能力

耦合在一起。

因此第一版建议采用口径 A，不强制遍历所有候选路径。

可以把当前 `selected_path` 明确视作：

- `PMP route oracle`
- 或 `route already decided by upper-layer routing`

这样最符合你现在的论文叙述。

## 5. 当前各模式是否公平

要分“现在已经跑出来的实验”与“后续模式选择实验”两种情况。

### 当前 `stk_dynamic_pmp` 实验的公平性

当前 STK 动态实验对以下对象是公平的：

- `LA-DP`
- `Greedy`
- `Uniform`
- `GS-Only`
- `Random`
- `GA`

原因是：

- 它们共享同一个时间片。
- 共享同一个 `selected_path`。
- 共享同一份链路带宽采样。
- 共享同一份任务规格。

所以它公平地回答的是：

- 在同一条已给定 PMP 路径上，不同算法谁更好。

但它不公平地回答不了下面这个问题：

- 在同一时间片场景下，不同推理模式谁更好。

因为现在的 `GS-Only` 仍然是在同一条 PMP 路径语境下得到的，不是“为 GS-Only 单独找一条最优路径”。

### 后续模式选择实验的公平性定义

模式选择实验里，公平不等于“大家走同一条路”。

更合理的公平标准是：

- 同一个时间片。
- 同一份 STK 可见性约束。
- 同一份链路采样。
- 同一份任务规格。
- 每个模式都可以在这个场景里使用“属于自己定义的最优执行结构”。

因此：

- `PMP` 可以有自己的既定线性路径。
- `GS-Only` 可以有自己最适合回传 GS 的路径。
- `Sat-Only` 可以有自己最适合的单星和对应入路/出路。
- `CDP` 可以有自己的 worker 集合和各自回传路径。

这样才是模式之间真正公平。

## 6. 模式选择第一版的建议边界

建议第一版先冻结以下边界：

- `PMP`：沿既定路径运行 `LA-DP`。
- `GS-Only`：单独选一条最快到 GS 的路径，在 GS 完成全推理。
- `Sat-Only`：选一个满足内存约束且预测总时延最小的单星方案。
- `CDP`：`RS -> 多 worker 分发 -> worker 各自推理 -> 各自直接回 GS`。
- 不引入星上聚合器。

`FWMS` 第一版不要直接操作原始路径，而是读取每个模式的最终评估结果，再做模式选择。

## 7. 后续接入实物实验，是否要新增实验入口

大概率要。

### 为什么不建议硬塞进现有入口

当前仓库里：

- `experiments_runner.py` 主要服务传统单拓扑实验。
- `stk_dynamic_experiment.py` 主要服务 STK 动态 PMP 理论实验。

如果未来把：

- 模式选择
- STK 时间片
- 代表性实物验证

全部硬塞进同一个入口，脚本会很快变得难维护。

### 更推荐的方式

保留现有职责分层，再新增一条“模式选择验证入口”，例如：

- `mode_selection_experiment.py`
  - 负责理论模式选择全量运行。
- `mode_selection_physical_verify.py`
  - 负责从理论结果中抽代表性 slot 和代表性模式，做少量 Jetson 实物验证。

这样分层更清楚：

- 理论全量：跑很多时间片。
- 实物验证：只抽少数代表场景验证趋势。

### 实物验证不建议做什么

不建议一上来做“所有时间片、所有模式、所有模型”的全矩阵实物验证。

更现实的做法是：

- 先用理论结果筛 3 到 5 个代表性时间片。
- 每类模式选 1 到 2 个代表配置。
- 用 Jetson 只验证时延趋势是否一致。

## 8. 现在最推荐的落地顺序

1. 新增模式选择入口，不改老主链。
2. `slot_scene` 先统一。
3. 第一版 `PMP` 直接复用 `selected_path`，不强制做候选路由枚举。
4. 完成 `GS-Only / Sat-Only / CDP` 的理论评估。
5. 接入 `FWMS`。
6. 再挑代表性场景做实物验证入口。
7. 最后统一重画图。

## 9. 一句话总结

当前仓库已经有：

- 传统理论实验入口
- 少量实物/混合实验入口
- STK 动态 PMP 入口
- 统一画图入口

但还没有：

- 面向模式选择的实物验证入口

当前已经有独立的 `mode_selection_experiment.py` 模式选择主链。接下来的正确方向不是继续堆在旧脚本上，而是保持 `PMP/STK` 主链和模式选择主链并行维护，再补一个面向模式选择的代表性实物验证入口。

## 10. Stage 6 已落地：slot_scene + PMP + GS-Only + Sat-Only + CDP + FWMS-Feature + Oracle-Min-Latency

当前已经新增模式选择入口：

```powershell
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_stage6_feature_oracle_b64 --batch-size-override 64
```

如果只是快速验证前两个时间片：

```powershell
python mode_selection_experiment.py --stk-run-dir result\stk_dynamic\stk_dynamic_yolo_001 --run-id mode_selection_yolo_stage6_feature_smoke --limit-slots 2 --batch-size-override 64
```

跨模型模式选择结果汇总已经归档到：

```text
result/mode_selection/cross_model_stage6_feature_oracle/
```

当前 Stage 6 做的事情：

- 从已有 STK 动态实验目录读取 `metadata.json`、`stk_dynamic_slots.csv`、`candidates/*.json` 和 `configs/*.json`。
- 只加载 `status=completed` 的时间片。
- 为每个时间片生成统一的 `slot_scene` JSON。
- 在该时间片的 `selected_path` 上运行 `PMP / LA-DP`。
- 为 `GS-Only` 重建带链路指标的 STK 候选路径，并选择预测总时延最低的 `RS -> ... -> GS` 路径。
- 如果 `GS-Only` 选中的路径和 `PMP selected_path` 相同，则复用原配置；如果不同，则生成独立的 `GS-Only network_config`。
- 为 `Sat-Only` 枚举候选路径中的单颗计算卫星，评估“入路传原始输入 + 单星全模型推理 + 出路回传最终结果”，并选预测总时延最低的单星方案。
- 为无聚合器版 `CDP` 枚举 worker 集合，使用 `LAWA-Discrete` 做 batch 离散分配，评估 `RS -> 多 worker 分发 -> worker 各自全模型推理 -> 各自直接回 GS`。
- 在四种基础模式之上同时运行两个选择器：`FWMS-Feature` 是论文式“可行性门控 + 特征加权”的模式边界判别；`Oracle-Min-Latency` 是所有可行基础模式中预测时延最低的上界基线。
- 输出统一的模式结果表 `slot_mode_results.csv`。
- 输出 `summary_by_mode.csv` 和 `fwms_selection_distribution.csv`。

输出目录默认是：

```text
result/mode_selection/<run_id>/
```

主要产物：

- `metadata.json`：记录本次模式选择实验阶段、来源 STK run、已实现模式和待实现模式。
- `scenes/<slot_id>_scene.json`：每个时间片统一场景对象。
- `configs/*_gs_only_network_config.json`：当 `GS-Only` 路径不同于 PMP 路径时生成的独立配置。
- `configs/*_sat_only_network_config.json`：`Sat-Only` 最优单星方案对应的独立配置。
- `configs/*_cdp_network_config.json`：无聚合器 CDP 的 worker、分发路由、回传路由和离散 batch 分配记录。
- `data/slot_mode_results.csv`：模式级结果长表，目前包含 `PMP / LA-DP`、`GS-Only / Min-Latency-Route`、`Sat-Only / Min-Latency-Single-Sat`、`CDP / LAWA-Discrete`、`FWMS-Feature / Feature-Weighted` 和 `Oracle-Min-Latency / Prediction-Min-Latency`。
- `data/summary_by_mode.csv`：按模式统计可行率、平均时延和平均卫星能耗。
- `data/fwms_selection_distribution.csv`：统计 `FWMS-Feature` 和 `Oracle-Min-Latency` 在所有时间片中选择各模式的比例。

当前 Stage 6 的边界：

- `PMP` 使用 `selected_path`，即论文中的“路由已给定”假设。
- 当前还不枚举所有 PMP 候选路径。
- `GS-Only` 使用自己的预测最低时延路径，在 GS 完成全模型推理。
- `Sat-Only` 只选择单颗卫星完成全模型推理，不做流水线切分，也不做星上聚合。
- `CDP` 不引入星上聚合器；每个 worker 都独立完成全模型推理，并沿自己的回传路由直接回 GS。
- `CDP` 使用离散样本分配，因此建议论文主结果优先使用 `--batch-size-override 64` 或更大的 batch 设置。
- `FWMS-Feature` 当前是可解释的特征加权边界判别器，不应表述为最低时延 Oracle。
- `Oracle-Min-Latency` 是预测最小时延上界基线，用来说明“如果所有模式都完整评估，最低时延会落在哪个模式”。
- 当前理论 CDP 已是无聚合器版，但实物侧 CDP 自动化还未完全对齐该口径。

## 11. 当前模式选择补强实验已经完成

这四件事已经完成：

1. `FWMS-Feature` 和 `Oracle-Min-Latency` 已在代码和结果表中区分清楚。
2. 跨模型模式边界结论表已输出到 `result/mode_selection/cross_model_stage6_feature_oracle/mode_boundary_conclusion_table.md`。
3. 可行率/完成率对比表已输出到 `result/mode_selection/cross_model_stage6_feature_oracle/mode_feasibility_completion_table.md`。
4. 实验叙事已更新为“稳定性与适用边界”，说明文档见 `doc/mode_selection_boundary_and_stability_summary.md`。

额外补强也已完成：

- CDP batch/worker 敏感性实验：`result/mode_selection/cdp_sensitivity_yolo_stage6/`。
- FWMS 权重敏感性实验：`result/mode_selection/fwms_weight_sensitivity_stage6/`。

当前最稳的论文叙事是：

```text
CDP 可行时通常给出低时延上界；但在大模型、内存压力或过大 batch 下会失效。
PMP 的价值不是总能低于 CDP，而是提供模型切分和星上协作保底。
FWMS 的价值是用可行性门控和任务特征判断模式边界，避免固定模式在动态拓扑和异构资源下失效。
```
