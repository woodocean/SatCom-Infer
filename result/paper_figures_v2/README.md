# paper_figures_v2

这个文件夹用于统一存放论文最终采用的图、实验参数说明，以及结题验收时的重跑步骤。

建议使用方式：

- 每个实验一个子文件夹，文件名按 `01_... / 02_... / 03_...` 排列。
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

常用命令如下。

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

- `plot-legacy`：传统 PMP 图
- `plot-mode-summary`：模式选择跨模型汇总图
- `plot-paper`：论文总图重画
- `plot-sensitivity`：敏感性实验图
- `plot-stk-summary`：STK 跨模型 PMP 汇总图

### 4. 半实物入口

```powershell
python thesis_entry.py semi-physical ...
python thesis_entry.py physical-orchestrator ...
```

- `semi-physical`：半实物验证
- `physical-orchestrator`：Jetson/PC 编排入口

### 5. 本次验收最常用命令

实验 1 重跑：

```powershell
python thesis_entry.py exp01 `
  --slot-id slot_033_064500_065000 `
  --models yolov5,vgg19,swin_base,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness
```

### 6. 验收时推荐只看这三个位置

- `result/paper_figures_v2/README.md`
- `result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/`
- `doc/acceptance_structure_and_commands.md`

---

## 实验 1：LADP-PMP 模式的算法有效性实验

### 图文件

- 正式图 PNG：[01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.png](</e:/Workspace/Python_pycharm_workspace/Projects/Collabrative_Inference/Neurosurgeon-main/result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.png>)
- 正式图 PDF：[01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.pdf](</e:/Workspace/Python_pycharm_workspace/Projects/Collabrative_Inference/Neurosurgeon-main/result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.pdf>)
- 汇总数据：[01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv](</e:/Workspace/Python_pycharm_workspace/Projects/Collabrative_Inference/Neurosurgeon-main/result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv>)
- 完整长表：[01_ladp_pmp_algorithm_effectiveness/slot_033_064500_065000_pmp_latency_norm_rerun_no_resnet_long.csv](</e:/Workspace/Python_pycharm_workspace/Projects/Collabrative_Inference/Neurosurgeon-main/result/paper_figures_v2/01_ladp_pmp_algorithm_effectiveness/slot_033_064500_065000_pmp_latency_norm_rerun_no_resnet_long.csv>)

### 实验目的

在固定 STK 时间片与固定链路条件下，对比 PMP 模式下不同算法的端到端时延，验证 `LADP` 在典型模型上的切分效果，并用 `ViT-Huge` 展示 PMP 的适用边界。

### 图口径

- 图类型：柱状图
- 横坐标：模型
- 纵坐标：归一化时延，`GS-Only = 1`
- 模型：`YOLOv5 / VGG19 / Swin-Base / ViT-Huge`
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
- 公共可见持续时间：`81.991 s`

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

- `YOLOv5`：`batch = 32`，输入尺寸 `640 x 640`
- `VGG19`：`batch = 32`，输入尺寸 `224 x 224`
- `Swin-Base`：`batch = 32`，输入尺寸 `224 x 224`
- `ViT-Huge`：`batch = 32`，输入尺寸 `224 x 224`

#### 5. 随机算法重跑口径

- `LADP / Greedy / Uniform / GS-Only`：各运行 `1` 次
- `GA / Random`：各重跑 `100` 次后取均值
- `GA` 参数：`pop_size = 20`，`generations = 200`，`mutation_rate = 0.2`
- 当前 `GA` 已允许空分段，因此可表达“纯中继节点/空字段”解

### 当前结果摘要

- `YOLOv5`：`LADP = 0.352`，`Greedy = 0.360`，`GA = 0.432`
- `VGG19`：`LADP = 0.282`，`GA = 0.284`
- `Swin-Base`：`LADP = 0.750`，`GA = 0.752`
- `ViT-Huge`：`LADP = 1.000`，`Greedy = 1.000`，`GA = 3.149`

可直接引用的结论：

- `YOLOv5` 和 `Swin-Base` 上，`LADP` 优于 `Greedy / Random / Uniform`。
- `VGG19` 上，`LADP` 与改进后的 `GA` 接近，说明该场景搜索空间相对简单。
- `ViT-Huge` 上，`LADP` 退化为 `GS-Only`，说明 PMP 对特征压缩不明显的大模型存在适用边界。

### 验收时如何重跑

在项目根目录下执行：

```powershell
conda activate satellite-split
python thesis_entry.py exp01 `
  --slot-id slot_033_064500_065000 `
  --models yolov5,vgg19,swin_base,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness
```

### 验收时看什么

1. 命令正常结束，终端最后输出：
   `result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness`
2. 检查图文件是否生成：
   - `exp01_ladp_pmp_algorithm_effectiveness.png`
   - `exp01_ladp_pmp_algorithm_effectiveness.pdf`
3. 检查汇总表是否生成：
   - `exp01_ladp_pmp_algorithm_effectiveness_summary.csv`
4. 打开图后重点看三点：
   - `YOLOv5` 上 `LADP` 明显优于 `GA / Random / Uniform`
   - `Swin-Base` 上 `LADP` 仍优于 `Greedy / Random / Uniform`
   - `ViT-Huge` 上 `LADP` 退化到 `GS-Only`

---

## 实验 2：LADP-PMP 模式的节点数量敏感性实验

- 图文件：待补充
- 实验目的：在同构算力条件下排除“任务集中到最强卫星”的影响，观察中继 LEO 数量变化对 `LADP` 的影响
- 图口径：
  - 图类型：折线图
  - 横坐标：中继 LEO 卫星数量
  - 纵坐标：归一化时延，`GS-Only = 1`
  - 模型：`YOLOv5 / VGG19`
  - 算法：`LADP / GA / Greedy / Random / Uniform / GS-Only`
- 固定参数：
  - `ISL = 5 Gbps`
  - `SGL = 100 Mbps`
  - `LEO` 同构算力：`3 TFLOPS`
  - `LEO` 内存：`4 GB`
  - `YOLOv5`：`batch = 32`，输入 `640 x 640`
  - `VGG19`：`batch = 32`，输入 `224 x 224`
  - 拓扑：链式 `RS -> SAT-* -> GS`
- 扫描变量：
  - 中继 `LEO` 数量：`1 / 2 / 3 / 4 / 5`
- 预期结论：
  - 节点数量增加会扩大切分搜索空间，但时延收益不一定线性提升
  - `LADP` 能避免无效节点带来的额外通信开销
- 重跑命令：待补充
- 验收步骤：待补充

## 实验 3：待补充

- 图文件：待补充
- 参数口径：待补充
- 重跑命令：待补充
- 验收步骤：待补充

## 实验 4：待补充

- 图文件：待补充
- 参数口径：待补充
- 重跑命令：待补充
- 验收步骤：待补充

## 实验 5：待补充

- 图文件：待补充
- 参数口径：待补充
- 重跑命令：待补充
- 验收步骤：待补充

## 实验 6：待补充

- 图文件：待补充
- 参数口径：待补充
- 重跑命令：待补充
- 验收步骤：待补充
