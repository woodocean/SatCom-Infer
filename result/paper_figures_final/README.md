# paper_figures_final

这个文件夹存放当前最终交付版论文图、对应汇总数据，以及一份按严格验收口径整理的实验说明。

这份说明不只回答“怎么跑”，还回答四个更关键的问题：

1. 每个实验到底在比较什么。
2. 哪些变量被固定，哪些变量被改变。
3. 图里观察到的现象，究竟能支持哪些结论。
4. 哪些结论其实不能说太满，否则容易被老师追问击穿。

## 统一重跑入口

根目录统一入口：

```powershell
python thesis_entry.py <command> ...
```

当前最终图对应命令：

```powershell
python thesis_entry.py exp01 --slot-id slot_033_064500_065000 --models yolov5,resnet101,vgg19,vit_huge --repeats 100 --out-dir result\paper_figures_final\01_ladp_pmp_algorithm_effectiveness
python thesis_entry.py exp02 --repeats 30 --out-dir result\paper_figures_final\02_ladp_pmp_node_count_sensitivity
python thesis_entry.py exp03 --models resnet101 --out-dir result\paper_figures_final\03_lawa_cdp_data_sensitivity
python thesis_entry.py exp04 --models yolov5 --out-dir result\paper_figures_final\04_lawa_cdp_worker_count_sensitivity
python thesis_entry.py exp05 --out-dir result\paper_figures_final\05_fwms_mode_selection_effectiveness
python thesis_entry.py exp06 --out-dir result\paper_figures_final\06_fwms_data_sensitivity
```

## 验收总原则

这 6 个实验并不是同一层级的问题，验收时不要混着讲：

- 实验1和实验2回答的是：`PMP` 模式本身值不值得做，`LADP` 是否比其他切分算法更合理。
- 实验3和实验4回答的是：`CDP` 模式本身值不值得做，`LAWA` 是否比简单分配更合理。
- 实验5和实验6回答的是：在 `PMP / CDP / GS-Only / Sat-Only / FWMS` 之间，模式选择是否合理，以及这种选择如何随任务变化而变化。

这意味着：

- 不能用实验1/2直接证明 `FWMS` 一定最优。
- 不能用实验3/4直接证明 `PMP` 一定不如 `CDP`。
- 不能把实验5/6里某一个模型的现象，直接推广成“所有模型都如此”。

## 实验 1：LADP-PMP 算法有效性

### 目的

在固定单个 STK 时间片、固定单条多跳路由、固定节点资源的条件下，对比 `LADP / Greedy / GA / Random / Uniform / GS-Only`，验证 `LADP` 在 `PMP` 模式中的切分有效性。

### 自变量

- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 算法：`LADP / Greedy / GA / Random / Uniform / GS-Only`

### 固定变量

- 时间片：`slot_033_064500_065000`
- 路由：固定 6 跳链路 `RS -> SAT-01 -> SAT-02 -> SAT-03 -> SAT-04 -> SAT-05 -> GS`
- 节点资源：5 颗 LEO + 1 个 GS 的算力与内存固定
- 输入批次：四个模型统一为 `batch=32`
- 链路带宽和传播时延：固定为该 slot 对应 STK 导出结果
- `GA / Random` 重复 `100` 次，其余算法为确定性单次求解

### 当前观察到的现象

- `YOLOv5`：`LADP=0.0338`，`GA=0.0338`，`Greedy=0.0945`，`Uniform=0.5937`
- `ResNet101`：`LADP=0.2885`，`GA=0.3264`，`Greedy=0.4210`
- `VGG19`：`LADP=0.2665`，`GA=0.2674`，`Greedy=0.5582`
- `ViT-Huge`：`LADP=1.0`，`Greedy=1.0`，`GS-Only=1.0`

### 可以支持的结论

- 在这个固定多跳动态场景下，`LADP` 对 `YOLOv5 / ResNet101 / VGG19` 的确优于 `Greedy / Random / Uniform`。
- `YOLOv5` 和 `VGG19` 上，`LADP` 与改进后的 `GA` 很接近，说明这两个场景的最优切分结构相对清晰。
- `ViT-Huge` 在当前资源和链路条件下退化为 `GS-Only`，说明 `PMP` 存在明确适用边界。

### 不能说太满的结论

- 不能说“LADP 普遍优于 GA”。这里只能说：在这个 slot 和这组模型上，`LADP` 至少不差于 GA，且求解更稳定。
- 不能说“ViT 一定不适合 PMP”。更准确的说法是：在当前 `batch=32`、当前路由、当前算力和内存约束下，`ViT-Huge` 不适合 PMP。
- 不能说“实验1证明了 PMP 优于 GS-Only”。实验1只证明了某些模型在该固定 slot 中，合理切分的 `PMP` 可以显著优于 `GS-Only`。

### 最危险的追问

- 为什么只选一个时间片？这个时间片是不是挑出来的有利样本？
- 为什么 `GA` 只用 `100` 次采样，你怎么知道没有更好的随机结果？
- 为什么四个模型都固定 `batch=32`，这会不会偏向 PMP？
- 为什么 `ViT-Huge` 的失败可以归因于模型特征，而不是当前卫星内存太小？

## 实验 2：LADP-PMP 节点数量敏感性

### 目的

在受控理论场景中，考察中继 LEO 数量变化对 `LADP` 的影响，并验证这种影响是否依赖资源异构性。

### 自变量

- 中继 LEO 数量：`1 / 2 / 3 / 4 / 5`
- 资源场景：`同构 / 单一异构 / 典型异构均值`
- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`

### 固定变量

- 基准路由模板：由实验1对应 slot 裁剪而来
- `ISL = 5 Gbps`
- `GSL = 100 Mbps`
- `GS = 300 TFLOPS, 64 GB`
- LEO 内存固定 `4 GB`
- 各模型仍使用 `batch=32`
- 公平口径：节点数变化时，总体算力预算按节点数同步扩展；异构模板按节点数归一化

### 当前观察到的现象

- `YOLOv5`：同构场景几乎不变，`0.0606 -> 0.0599`
- `YOLOv5`：异构场景反而变差，`0.0606 -> 0.0904`
- `ResNet101`：典型异构均值下降，`0.3566 -> 0.2842`
- `VGG19`：典型异构均值明显下降，`0.5769 -> 0.3489`
- `ViT-Huge`：三种场景都保持 `1.0`

### 可以支持的结论

- 节点数量增加本身不保证更优，收益依赖模型压缩特征与资源异构结构。
- 对 `VGG19` 这类可在中间层显著压缩数据的模型，更多节点和异构资源会扩大 `PMP` 的收益空间。
- 对 `YOLOv5`，额外节点在某些异构模板下可能引入更多通信负担而不是收益。

### 不能说太满的结论

- 不能说“节点越多越差”或“节点越多越好”。这张图恰恰说明收益方向依赖模型。
- 不能说这是 STK 动态场景结论。实验2是受控理论实验，不是多时间片真实动态平均。
- 不能说“异构一定优于同构”。这里只能说：在某些模型上，异构给了 `LADP` 更多可利用空间。

### 最危险的追问

- 总算力口径到底是“固定总算力”还是“随节点数扩展总算力”？如果是后者，这还是纯节点数敏感性吗？
- 路由是从一个 slot 裁剪来的，是否会把实验1里的拓扑偏好带进实验2？
- 为什么只扫到 5 个节点？再多会不会趋势反转？
- `YOLOv5` 在异构场景下变差，说明 `LADP` 不稳，还是说明实验设置本身偏向通信瓶颈？

## 实验 3：LAWA-CDP 数据量敏感性

### 目的

在固定 worker 数量下，比较输入数据量变化时 `LAWA` 与简单分配策略在 `CDP` 模式下的表现。

### 自变量

- 输入数据量：`16 / 32 / 64 / 128`
- worker 场景：`homogeneous / heterogeneous`
- 算法：`LAWA / Greedy / Uniform / Random / Sat-Only`

### 固定变量

- 最终图只保留 `ResNet101`
- worker 数量固定 `4`
- profile：`config/dnn_profiles_database_jetson.json`
- `ResNet101` 输入尺寸固定 `224x224`
- 随机基线固定种子，重复 `30` 次

### 当前观察到的现象

在 `heterogeneous` 场景、`ResNet101` 上：

- `LAWA`：`0.4949 -> 0.4072`
- `Greedy`：约 `0.4823 -> 0.4803`
- `Uniform`：约 `0.9688 -> 0.9664`
- `Random`：约 `1.0222 -> 1.0894`

### 可以支持的结论

- 在异构 worker 场景下，`LAWA` 能随着数据量增大更好地利用强 worker，整体优于 `Uniform / Random`。
- `Greedy` 在小规模下可能接近 `LAWA`，但随着输入数据量增大，`LAWA` 的优势更稳定。
- `CDP` 的收益不仅来自“多星并行”，更来自“异构感知的数据量分配”。

### 不能说太满的结论

- 不能说 `LAWA` 在所有 batch 都明显优于 `Greedy`。这里 `Greedy` 在小 batch 下很接近。
- 不能说实验3证明了 `CDP` 比 `PMP` 更适合 `ResNet101`。实验3压根没和 `PMP` 做同口径对比。
- 不能说“输入越大，LAWA 优势越大”是普适规律。这里只保留了 `ResNet101`。

### 最危险的追问

- 为什么只展示 `ResNet101`，是不是因为别的模型现象不够漂亮？
- 既然 `Greedy` 和 `LAWA` 在小 batch 下很接近，你怎么证明 LAWA 的复杂度是值得的？
- 为什么归一化基准选 `Sat-Only` 而不是 `GS-Only`？
- 这个实验没有动态路由，能不能代表真实星地环境下的 CDP？

## 实验 4：LAWA-CDP worker 数量敏感性

### 目的

在固定输入数据量下，比较 worker 数量增加时 `LAWA` 的收益是否持续，以及是否优于简单贪心。

### 自变量

- worker 数量：`1 / 2 / 3 / 4 / 5`
- 算法：`LAWA / Greedy / Sat-Only`

### 固定变量

- 最终图只保留 `YOLOv5`
- 输入数据量固定 `64`
- profile：`config/dnn_profiles_database_jetson.json`
- worker 场景：典型异构 worker 集合逐步增加

### 当前观察到的现象

- `LAWA`：`1.0000 -> 0.3767`
- `Greedy`：`1.0000 -> 0.4304`
- `Sat-Only`：恒为 `1.0`

### 可以支持的结论

- 在当前异构 worker 配置下，worker 数量增加能明显提升 `YOLOv5` 的 `CDP` 收益。
- `LAWA` 全程优于 `Greedy`，说明异构感知分配在多 worker 条件下有稳定价值。
- 收益并不是线性叠加：从 `1->2`、`2->3`、`3->4`、`4->5` 的边际改善不同。

### 不能说太满的结论

- 不能说“worker 越多越好”是一般规律。这里只展示 `YOLOv5`，且 worker 是一组特定异构配置。
- 不能说 `LAWA` 对所有模型都有同样幅度收益。`ResNet101 / VGG19` 当前未纳入最终图。
- 不能说去掉 `Uniform / Random` 之后它们就不重要。这里只是最终图聚焦，不代表它们没有分析价值。

### 最危险的追问

- 为什么最终图只保留 `YOLOv5`？这会不会有选择性展示的问题？
- worker 增加时，总带宽、总算力、总回传开销是不是也在变？如果都在变，这还是单一变量实验吗？
- `Sat-Only` 被固定为 1 的归一化口径，会不会掩盖绝对时延上的某些反常点？

## 实验 5：FWMS 模式选择有效性

### 目的

在同一批真实 STK 时间片上，对 `PMP / CDP / GS-Only / Sat-Only / FWMS` 做公平比较，验证 `FWMS` 的模式选择是否符合各模型任务特征。

### 自变量

- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 模式：`PMP / CDP / GS-Only / Sat-Only / FWMS`

### 固定变量

- 输入数据量固定 `64`
- 共同时间片集合：`28` 个 common slots
- 公平筛选：
  - `PMP` 共享路由至少包含 `3` 颗 `LEO`
  - `CDP active_sat_count >= 3`
  - `PMP / GS-Only / Sat-Only` 在每个 slot 上共用同一路由
- `CDP worker_count = 4`
- `worker_memory = 2048 MB`
- `gsl_bandwidth = 100 Mbps`
- `gs_compute_factor = 100`
- profile：`config/dnn_profiles_database_jetson.json`

### 当前观察到的现象

- `YOLOv5`：`CDP=0.0278`，`FWMS=0.0286`，`PMP=0.0986`
- `ResNet101`：`CDP=0.1751`，`FWMS=0.1922`，`PMP=0.6340`
- `VGG19`：`CDP` 不可行，`Sat-Only` 不可行，`FWMS=PMP=0.9138`
- `ViT-Huge`：`CDP` 不可行，`Sat-Only` 不可行，`FWMS=PMP=GS-Only=1.0`

### 可以支持的结论

- 在当前共同 slot 平均口径下，`FWMS` 没有简单地固定偏向 `PMP` 或 `GS-Only`，而是随模型可行性与收益空间变化。
- `YOLOv5` 与 `ResNet101` 上，`CDP` 明显优于 `PMP`，说明当前新 profile 下，这两个模型的主导瓶颈更接近“分摊计算”而非“流水线压缩回传”。
- `VGG19 / ViT-Huge` 的 `CDP` 与 `Sat-Only` 在当前内存约束下不可行，`FWMS` 合理回退到可行模式。

### 不能说太满的结论

- 不能说 `FWMS` 已经达到 oracle 最优。这里只能说它给出了合理、可解释、可行的模式选择。
- 不能说 `YOLOv5` 一定应该选 `CDP`。这是基于当前 `batch=64`、`worker=4`、`2 GB` 内存和共同 slot 集合的结果。
- 不能说 `VGG19 / ViT-Huge` 天然不适合 `CDP`。更准确地说：在当前 worker 内存与共同 slot 条件下不可行。

### 最危险的追问

- 为什么 `FWMS` 比 `CDP` 略差？如果最终目的是真正最小时延，为什么不直接选 `CDP`？
- 公平性到底如何保证？为什么 PMP、GS-Only、Sat-Only 必须共路由？
- `VGG19 / ViT-Huge` 的不可行是算法问题、内存问题，还是你们人为设置太严？
- `28` 个 common slots 会不会过滤掉了大量对 `CDP` 不利或对 `PMP` 不利的时间片？

## 实验 6：FWMS 输入数据量敏感性

### 目的

在与实验5相同的公平模式比较框架下，只改变 `YOLOv5` 的输入数据量，考察模式收益随 batch 增大如何变化。

### 自变量

- 输入数据量：`16 / 32 / 64 / 128`
- 模式：`PMP / CDP / GS-Only / Sat-Only / FWMS`

### 固定变量

- 模型固定 `YOLOv5`
- 共同时间片集合：同样为 `28` 个 common slots
- 路由公平口径与实验5一致
- `CDP worker_count = 4`
- `worker_memory = 2048 MB`
- `gsl_bandwidth = 100 Mbps`
- `gs_compute_factor = 100`

### 当前观察到的现象

- `GS-Only`：`3203.63 -> 25269.02 ms`，近似线性增长
- `PMP`：归一化时延稳定在 `0.0986 ~ 0.1149`
- `CDP`：`0.0399 -> 0.0278`，到 `batch=128` 不可行
- `Sat-Only`：与 `PMP` 基本重合，到 `batch=128` 不可行
- `FWMS`：`batch=16` 取值 `0.0587`，`32/64` 接近 `CDP`，`128` 回退到 `PMP`

### 可以支持的结论

- 在当前公平口径下，`GS-Only` 的主要代价随输入规模近似线性上升。
- `CDP` 在 `batch=16/32/64` 上明显优于 `PMP`，但到 `batch=128` 受到可行性约束。
- `FWMS` 体现出“可行时选低时延，不可行时回退保底”的策略行为。

### 不能说太满的结论

- 不能说 `FWMS` 会随着 batch 单调变好或变差。它是离散模式选择，可能出现阈值切换。
- 不能说 `PMP` 对 `YOLOv5` 没价值。它在 `batch=128` 时成为可行保底，而且比 `GS-Only` 低很多。
- 不能说 `CDP` 在大 batch 上绝对更强。当前 `2 GB` worker 内存让 `batch=128` 不可行。

### 最危险的追问

- 为什么 `GS-Only` 现在看起来线性增长，而你前面说它“不一定线性”？是不是口径改过？
- `FWMS` 在 `batch=16` 没直接选最优 `CDP`，这个判据是不是太保守？
- `batch=128` 时 `CDP` 不可行，到底是内存限制还是路由限制？
- 为什么只展示 `YOLOv5`？如果换成 `ResNet101`，趋势还会这样吗？

## 答辩时必须主动交代的三件事

### 1. 哪些实验是“真实动态平均”，哪些只是“受控理论实验”

- 实验1：单 slot 真实路由实例
- 实验2：受控理论扫描
- 实验3/4：受控 CDP 资源实验
- 实验5/6：多 slot 公平平均

### 2. 哪些结论是“算法优劣”，哪些只是“适用边界”

- `LADP` 和 `LAWA` 的图主要讲算法优劣
- `ViT-Huge`、`VGG19` 的不可行更多讲适用边界与资源约束

### 3. 当前最脆弱的地方

- 实验1只看一个 slot，外推能力有限
- 实验2不是 STK 动态平均
- 实验3/4只保留一个模型做最终图，存在“展示压缩”风险
- 实验5/6仍是 profile 驱动的仿真，不是完整半实物闭环

## 配套文档

- [STRICT_ACCEPTANCE_CHECKLIST.md](E:\Workspace\Python_pycharm_workspace\Projects\Collabrative_Inference\Neurosurgeon-main\result\paper_figures_final\STRICT_ACCEPTANCE_CHECKLIST.md)

