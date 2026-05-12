# paper_figures_v2

这个文件夹统一存放论文最终采用的图、实验参数说明，以及结题验收时的重跑步骤。

建议使用方式：
- 每个实验一个子文件夹，文件夹名按 `01_... / 02_... / 03_...` 排列。
- 每个子文件夹至少包含：
  - 最终图 `png/pdf`
  - 汇总数据 `csv`
  - 如有需要，再放长表 `long.csv`
- 本文件统一记录：
  - 图放在哪里
  - 该实验的参数口径
  - 结题验收时怎么重跑

## 统一入口速查表

以后优先使用根目录统一入口：

```powershell
python thesis_entry.py <command> ...
```

### 1. 主实验入口

```powershell
python thesis_entry.py run-legacy ...
python thesis_entry.py run-stk ...
python thesis_entry.py run-mode ...
```

- `run-legacy`：传统 PMP 理论实验
- `run-stk`：STK 动态拓扑 PMP 实验
- `run-mode`：模式选择实验

### 2. 前处理入口

```powershell
python thesis_entry.py build-stk-config ...
```

- `build-stk-config`：从 STK 报告生成候选路径和 `network_config`

### 3. 绘图入口

```powershell
python thesis_entry.py plot-legacy ...
python thesis_entry.py plot-mode-summary ...
python thesis_entry.py plot-paper ...
python thesis_entry.py plot-sensitivity ...
python thesis_entry.py plot-stk-summary ...
```

### 4. 半实物入口

```powershell
python thesis_entry.py semi-physical ...
python thesis_entry.py physical-orchestrator ...
```

### 5. 本次验收最常用命令
实验 00 重画：

```powershell
python thesis_entry.py exp00 --profile config/dnn_profiles_database_pc.json --reference-profile config/dnn_profiles_database_jetson.json --batch-size 1 --out-dir result\paper_figures_v2\00_layer_output_distribution
```

实验 1 重跑：

```powershell
python thesis_entry.py exp01 -- `
  --slot-id slot_033_064500_065000 `
  --models yolov5,resnet101,vgg19,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness
```

实验 2 重跑：

```powershell
python thesis_entry.py exp02 `
  --repeats 30 `
  --out-dir result\paper_figures_v2\02_ladp_pmp_node_count_sensitivity
```

### 6. 验收时推荐只看这几个位置

- `result/paper_figures_v2/README.md`
- `result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/`
- `result/paper_figures_v2/02_ladp_pmp_node_count_sensitivity/`
- `doc/acceptance_structure_and_commands.md`

---


## 实验 00：层级输出特征图数据量分布

### 图文件

- [exp00_layer_output_distribution_yolov5.png](./00_layer_output_distribution/exp00_layer_output_distribution_yolov5.png)
- [exp00_layer_output_distribution_yolov5.pdf](./00_layer_output_distribution/exp00_layer_output_distribution_yolov5.pdf)
- [exp00_layer_output_distribution_resnet101.png](./00_layer_output_distribution/exp00_layer_output_distribution_resnet101.png)
- [exp00_layer_output_distribution_resnet101.pdf](./00_layer_output_distribution/exp00_layer_output_distribution_resnet101.pdf)
- [exp00_layer_output_distribution_vgg19.png](./00_layer_output_distribution/exp00_layer_output_distribution_vgg19.png)
- [exp00_layer_output_distribution_vgg19.pdf](./00_layer_output_distribution/exp00_layer_output_distribution_vgg19.pdf)
- [exp00_layer_output_distribution_vit_huge.png](./00_layer_output_distribution/exp00_layer_output_distribution_vit_huge.png)
- [exp00_layer_output_distribution_vit_huge.pdf](./00_layer_output_distribution/exp00_layer_output_distribution_vit_huge.pdf)
- [exp00_layer_output_distribution_summary.csv](./00_layer_output_distribution/exp00_layer_output_distribution_summary.csv)
- [exp00_profile_consistency_report.csv](./00_layer_output_distribution/exp00_profile_consistency_report.csv)

### 图口径

- profile：`config/dnn_profiles_database_pc.json`
- reference profile：`config/dnn_profiles_database_jetson.json`
- 批次：`b1`（YOLOv5 为 `640x640`，其余为 `224x224`；若 profile 未包含 b1，将按最小可用 batch 线性缩放）
- 绿色虚线：输入 `Input`
- 红色柱：层输出大于输入大小
- 蓝色柱：层输出小于或等于输入大小
- 最后一层标注 `Result`
- 重画前会检查 PC/Jetson 两个 profile 的四个模型层数和 `comm_total_mb` 是否一致

### 验收时如何重画

```powershell
python thesis_entry.py exp00 --profile config/dnn_profiles_database_pc.json --reference-profile config/dnn_profiles_database_jetson.json --batch-size 1 --out-dir result\paper_figures_v2\00_layer_output_distribution
```
## 实验 1：LADP-PMP 模式的算法有效性实验

### 图文件

- [exp01_ladp_pmp_algorithm_effectiveness.png](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.png)
- [exp01_ladp_pmp_algorithm_effectiveness.pdf](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.pdf)
- [exp01_ladp_pmp_algorithm_effectiveness_summary.csv](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv)
- [exp01_ladp_pmp_algorithm_effectiveness_long.csv](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_long.csv)

### 实验目的

在固定 STK 时间片与固定链路条件下，对比 PMP 模式下不同算法的端到端时延，验证 `LADP` 在典型模型上的切分效果，并用 `ViT-Huge` 展示 PMP 的适用边界。

### 图口径

- 图类型：柱状图
- 横坐标：模型
- 纵坐标：归一化时延，`GS-Only = 1`
- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 算法：`LADP / Greedy / GA / Random / Uniform / GS-Only`

### 固定参数

#### 1. STK 时间片与拓扑

- 时间片：`slot_033_064500_065000`
- 时间范围：`14 Apr 2026 06:45:00` 到 `06:50:00`
- 选中路径：`RS -> SAT-01 -> SAT-02 -> SAT-03 -> SAT-04 -> SAT-05 -> GS`
- 对应 STK 原路径：`RS -> LEO156 -> LEO052 -> LEO043 -> LEO125 -> LEO105 -> Shenzhen`
- 总跳数：`6`
- 总距离：`20294.505687 km`
- 总传播时延：`67.695184 ms`

#### 2. 链路参数

- `RS -> SAT-01`：`11642.3057 Mbps`，`12.134586 ms`
- `SAT-01 -> SAT-02`：`15464.1532 Mbps`，`10.715799 ms`
- `SAT-02 -> SAT-03`：`17968.7732 Mbps`，`16.938235 ms`
- `SAT-03 -> SAT-04`：`2207.1161 Mbps`，`9.878242 ms`
- `SAT-04 -> SAT-05`：`1169.4010 Mbps`，`13.932614 ms`
- `SAT-05 -> GS`：`116.9512 Mbps`，`4.095708 ms`

#### 3. 节点资源

- `SAT-01`：`4.106 TFLOPS`，`4096 MB`
- `SAT-02`：`9.201 TFLOPS`，`2048 MB`
- `SAT-03`：`4.597 TFLOPS`，`4096 MB`
- `SAT-04`：`1.867 TFLOPS`，`2048 MB`
- `SAT-05`：`4.799 TFLOPS`，`4096 MB`
- `GS`：`300 TFLOPS`，`64000 MB`

#### 4. 任务参数

- `YOLOv5`：`batch = 32`，输入 `640 x 640`
- `VGG19`：`batch = 32`，输入 `224 x 224`
- `ResNet101`：`batch = 32`，输入 `224 x 224`
- `ViT-Huge`：`batch = 32`，输入 `224 x 224`

#### 5. 随机算法口径

- `LADP / Greedy / Uniform / GS-Only`：各运行 `1` 次
- `GA / Random`：各重复 `100` 次后取均值
- `GA` 参数：`pop_size = 20`，`generations = 200`，`mutation_rate = 0.2`
- 当前 `GA` 已允许空分段，因此可表达“纯中继节点/空字段”解

### 当前结果摘要

- `YOLOv5`：`LADP = 0.035`，`Greedy = 0.097`，`GA = 0.035`
- `VGG19`：`LADP = 0.282`，`GA = 0.284`
- `ResNet101`：`LADP = 0.863`，`GA = 0.902`
- `ViT-Huge`：`LADP = 1.000`，`Greedy = 1.000`

可直接引用的结论：
- `YOLOv5 / ResNet101 / VGG19` 上，`LADP` 整体优于 `Greedy / Random / Uniform`
- `YOLOv5 / VGG19` 上，`LADP` 与改进后的 `GA` 接近，说明该场景搜索空间相对简单
- `ViT-Huge` 上，`LADP` 退化为 `GS-Only`，说明 PMP 对特征压缩不明显的大模型存在适用边界

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp01 -- `
  --slot-id slot_033_064500_065000 `
  --models yolov5,resnet101,vgg19,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness
```

### 验收时看什么

1. 命令正常结束，终端最后输出目标目录
2. 检查是否生成：
   - `exp01_ladp_pmp_algorithm_effectiveness.png`
   - `exp01_ladp_pmp_algorithm_effectiveness.pdf`
   - `exp01_ladp_pmp_algorithm_effectiveness_summary.csv`
3. 打开图后重点看三点：
   - `YOLOv5` 中 `LADP` 明显优于 `GA / Random / Uniform`
   - `ResNet101` 中 `LADP` 仍优于 `Greedy / Random / Uniform`
   - `ViT-Huge` 中 `LADP` 退化到 `GS-Only`

---

## 实验 2：LADP-PMP 模式的节点数量敏感性实验

### 图文件

- [exp02_ladp_pmp_node_count_sensitivity.png](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity.png)
- [exp02_ladp_pmp_node_count_sensitivity.pdf](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity.pdf)
- [exp02_ladp_pmp_node_count_sensitivity_active_sat_count.png](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_active_sat_count.png)
- [exp02_ladp_pmp_node_count_sensitivity_summary.csv](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_summary.csv)
- [exp02_ladp_pmp_node_count_sensitivity_scenarios.csv](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_scenarios.csv)
- [exp02_ladp_pmp_node_count_sensitivity_long.csv](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_long.csv)

### 实验目的

在固定平均算力预算下，对比同构、单一异构和典型异构均值场景中，中继 LEO 数量变化对 `LADP` 的影响，并观察最优解实际启用的计算卫星数量。

### 图口径

- 图类型：折线图
- 横坐标：中继 LEO 卫星数量
- 纵坐标：归一化时延，`GS-Only = 1`
- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 曲线：`同构场景 / 异构场景 / 典型异构均值`
- 额外图：`active_sat_count` 展示 LADP 实际启用的计算星数量

### 固定参数

- 基准拓扑：基于 `slot_033_064500_065000` 的 5 星链式路由裁剪得到
- `ISL = 5 Gbps`
- `SGL = 100 Mbps`
- `LEO` 同构算力：`3 TFLOPS`
- `LEO` 内存：`4 GB`
- `GS` 算力：`300 TFLOPS`
- `GS` 内存：`64 GB`
- `YOLOv5`：`batch = 32`，输入 `640 x 640`
- `ResNet101 / VGG19 / ViT-Huge`：`batch = 32`，输入 `224 x 224`
- 扫描变量：中继 `LEO` 数量 `1 / 2 / 3 / 4 / 5`
- 公平口径：任意节点数 `N` 下，总 LEO 算力均固定为 `3N TFLOPS`
- 异构模板会按当前节点数归一化，例如 `N=3` 时 `1,2,3,4,5` 会缩放为 `1.5,3.0,4.5`

### 当前结果摘要

- `YOLOv5`：同构场景变化很小，说明同构链路下最优策略接近单星完成；异构均值略升，主要来自额外跳数通信开销
- `ResNet101`：异构场景随节点数增加有一定收益，说明算力差异能带来计算/通信权衡空间
- `VGG19`：同构和异构场景均随节点数增加下降，且异构均值收益更明显，是 PMP 多星协同较清晰的案例
- `ViT-Huge`：基本退化为 `GS-Only`，说明该模型在当前参数下不适合 PMP 切分

可直接引用的结论：
- 增加可用中继节点会扩大模型切分搜索空间，但收益依赖模型特征和资源异构性
- 在同构算力、同构带宽下，最优解常倾向少量节点完成计算，更多节点不一定有利
- 在存在算力异构时，`LADP` 能自动权衡“强节点计算收益”和“额外通信开销”，部分模型会启用多颗计算星

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp02 `
  --out-dir result\paper_figures_v2\02_ladp_pmp_node_count_sensitivity
```

### 验收时看什么

1. 命令正常结束，终端最后输出目标目录
2. 检查是否生成：
   - `exp02_ladp_pmp_node_count_sensitivity.png`
   - `exp02_ladp_pmp_node_count_sensitivity.pdf`
   - `exp02_ladp_pmp_node_count_sensitivity_active_sat_count.png`
   - `exp02_ladp_pmp_node_count_sensitivity_summary.csv`
3. 打开图后重点看三点：
   - `node_count = 1` 时三类场景一致，说明同总算力公平口径生效
   - 同构场景收益有限，说明单任务 PMP 不会因为节点变多就自动变快
   - `VGG19` 等模型在异构场景中收益更明显，说明 LADP 的价值主要体现在资源异构下的切分权衡

---

## 实验 3：LAWA-CDP 模式的数据量敏感性实验

### 图文件

- [exp03_lawa_cdp_data_sensitivity_yolov5.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_yolov5.png)
- [exp03_lawa_cdp_data_sensitivity_resnet101.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_resnet101.png)
- [exp03_lawa_cdp_data_sensitivity_vgg19.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_vgg19.png)
- [exp03_lawa_cdp_data_sensitivity_summary.csv](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_summary.csv)
- [exp03_lawa_cdp_data_sensitivity_long.csv](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_long.csv)

### 图口径

- 图类型：折线图，三个模型分别出图，标题统一为 `LAWA-CDP 模式的数据量敏感性实验`
- 横坐标：输入数据量（样本数）`16 / 32 / 64 / 128`
- 纵坐标：归一化时延，`Sat-Only = 1`
- 模型：`YOLOv5 / ResNet101 / VGG19`
- 算法：`LAWA / 贪心 / 均匀 / 随机 / Sat-Only`
- 场景：同构 worker 场景；异构 worker 场景

### 固定参数

- 并行 worker 数量：`4`
- 同构 worker：算力、分发链路、回传链路均相同
- 异构 worker：算力、分发链路、回传链路存在差异
- 推理 profile：`config/dnn_profiles_database_jetson.json`
- 输入尺寸：`YOLOv5 = 640 x 640`，`ResNet101 / VGG19 = 224 x 224`
- 随机基线：固定随机种子，重复 `30` 次取均值

### 当前结论摘要

- 同构 worker 场景下，`LAWA / 均匀 / 贪心` 接近，说明 worker 能力一致时简单均分已经较合理。
- 异构 worker 场景下，随着输入数据量增大，`LAWA` 能更稳定地利用强算力和好链路 worker，通常优于均匀和随机。
- 输入数据量较小时，离散样本分配和传播开销占比更高，LAWA 的优势不一定明显；这可以作为 CDP/LAWA 的适用边界说明。

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp03 `
  --out-dir result\paper_figures_v2\03_lawa_cdp_data_sensitivity
```

---

## 实验 4：LAWA-CDP 模式的 worker 数量敏感性实验

### 图文件

- [exp04_lawa_cdp_worker_count_sensitivity_yolov5.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_yolov5.png)
- [exp04_lawa_cdp_worker_count_sensitivity_resnet101.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_resnet101.png)
- [exp04_lawa_cdp_worker_count_sensitivity_vgg19.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_vgg19.png)
- [exp04_lawa_cdp_worker_count_sensitivity_summary.csv](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_summary.csv)
- [exp04_lawa_cdp_worker_count_sensitivity_long.csv](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_long.csv)

### 图口径

- 图类型：折线图，三个模型分别出图，标题统一为 `LAWA-CDP 模式的 worker 数量敏感性实验`
- 横坐标：并行 worker 卫星数量 `1 / 2 / 3 / 4 / 5`
- 纵坐标：归一化时延，`Sat-Only = 1`
- 模型：`YOLOv5 / ResNet101 / VGG19`
- 算法：`LAWA / 贪心 / 均匀 / 随机 / Sat-Only`
- 场景：典型异构 worker 场景

### 固定参数

- 输入数据量：`64`
- 输入尺寸：`YOLOv5 = 640 x 640`，`ResNet101 / VGG19 = 224 x 224`
- worker 能力：随 worker 数量增加，加入具有不同算力和链路质量的 worker
- 推理 profile：`config/dnn_profiles_database_jetson.json`
- 随机基线：固定随机种子，重复 `30` 次取均值

### 当前结论摘要

- worker 数量增加通常能降低 CDP 时延，但收益不是线性的。
- 当新增 worker 链路或算力较弱时，均匀分配和随机分配可能引入额外负担。
- `LAWA` 会根据 worker 计算能力和链路状态调节数据量，因此在异构 worker 场景下比均匀和随机更稳定。

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp04 `
  --out-dir result\paper_figures_v2\04_lawa_cdp_worker_count_sensitivity
```

## 实验 5：FWMS 模式选择有效性实验

### 图文件

- [exp05_fwms_mode_selection_effectiveness.png](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness.png)
- [exp05_fwms_mode_selection_effectiveness.pdf](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness.pdf)
- [exp05_fwms_mode_selection_effectiveness_summary.csv](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness_summary.csv)

### 图口径

- 图类型：双子图柱状图
- 子图 1：理论仿真归一化时延
- 子图 2：半实物仿真占位，当前不填充数据
- 横坐标：模型
- 纵坐标：归一化时延，`GS-Only = 1`
- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 模式：`PMP / CDP / GS-Only / FWMS`

### 固定参数

- 输入数据量：`64`
- CDP worker 数量：`4`
- worker 内存：`2048 MB`
- GSL 带宽：`100 Mbps`
- GS 计算倍率：相对 Jetson profile 为 `100 x`
- PMP 参考：实验 1 的 `LADP` 归一化结果
- CDP 参考：异构 worker 场景下的 `LAWA`

### 当前结论摘要

- `YOLOv5` 修正最终检测输出后，PMP 时延最低，FWMS 选择 PMP，说明小最终输出任务适合流水线/星上压缩。
- `ResNet101 / VGG19` 中 CDP 可行且低时延，FWMS 选择 CDP。
- `ViT-Huge` 完整模型权重超过 `2 GB` worker 内存，CDP 不可行，FWMS 回退到 GS-Only/PMP 保底。
- 半实物子图目前明确标注待补充，不使用伪数据。

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp05 `
  --out-dir result\paper_figures_v2\05_fwms_mode_selection_effectiveness
```

---

## 实验 6：FWMS 输入数据量敏感性实验

### 图文件

- [exp06_fwms_data_sensitivity_yolov5.png](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5.png)
- [exp06_fwms_data_sensitivity_yolov5.pdf](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5.pdf)
- [exp06_fwms_data_sensitivity_yolov5_summary.csv](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5_summary.csv)

### 图口径

- 图类型：折线图
- 横坐标：输入数据量（样本数）`16 / 32 / 64 / 128`
- 纵坐标：平均端到端时延 / ms
- 模型：`YOLOv5`
- 模式：`PMP / CDP / GS-Only / FWMS`

### 固定参数

- CDP worker 数量：`4`
- worker 内存：`2048 MB`
- GSL 带宽：`100 Mbps`
- GS 计算倍率：相对 Jetson profile 为 `100 x`
- PMP 参考：实验 1 的 `LADP` 归一化结果随输入数据量线性缩放
- CDP 参考：异构 worker 场景下的 `LAWA`

### 当前结论摘要

- `YOLOv5` 最终检测结果很小，PMP 能显著降低回传数据量，因此随输入数据量增大仍保持低时延。
- CDP 相比 GS-Only 仍有明显优势，但在当前 YOLOv5 口径下不如 PMP。
- FWMS 在各输入数据量下均选择 PMP，说明模式选择不是固定偏向 CDP，而是受任务输出压缩特征影响。

### 验收时如何重跑

```powershell
conda activate satellite-split
python thesis_entry.py exp06 `
  --out-dir result\paper_figures_v2\06_fwms_data_sensitivity
```


