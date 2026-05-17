# paper_figures_final

本文件夹存放当前最终使用的 6 组实验图及对应汇总结果。

建议介绍顺序：

1. 实验 5、6：系统层模式比较与模式切换
2. 实验 1、2：`PMP` 模式与 `LADP`
3. 实验 3、4：`CDP` 模式与 `LAWA`

## 统一说明

- `PMP`：模型流水切分
- `CDP`：数据并行
- `GS-Only`：全部回传地面站推理
- `Sat-Only`：单星完成推理
- `FWMS`：模式选择

除实验 2、3、4 的受控扫描外，其余结果均使用当前仓库中的新 profile 与当前代码口径生成。

现场复现时，如只需要弹图、不保存结果，可直接使用下面的 `show-only` 命令。

## 实验 1：LADP-PMP 算法有效性

### 图文件

- `01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.png`

### 参数

- 时间片：`slot_033_064500_065000`
- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- batch：统一 `32`
- 算法：`LADP / Greedy / GA / Random / Uniform / GS-Only`
- `GA / Random`：各重复 `100` 次取均值
- 路由：固定该时间片对应的单条 STK 多跳路径

### 现象

- `YOLOv5`：`LADP` 与 `GA` 接近，明显优于 `Greedy / Random / Uniform`
- `ResNet101`：`LADP` 优于 `Greedy / Random / Uniform`
- `VGG19`：`LADP` 与 `GA` 接近，优于其余基线
- `ViT-Huge`：`LADP` 退化到 `GS-Only`

### 结论

- 在当前固定时间片与固定路由下，`LADP` 能为 `PMP` 提供更合理的层切分。
- 对 `YOLOv5 / ResNet101 / VGG19`，`PMP` 具有明显收益。
- `ViT-Huge` 在当前配置下不适合 `PMP`。

### 复现命令

```powershell
python -m tools.paper_figures.run_stk_slot_pmp_highlight --slot-id slot_033_064500_065000 --models yolov5,resnet101,vgg19,vit_huge --repeats 100 --show-only --out-dir result\.codex_tmp\exp01_preview
```

## 实验 2：LADP-PMP 节点数量敏感性

### 图文件

- `02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity.png`

### 参数

- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- batch：统一 `32`
- 中继 LEO 数量：`1 / 2 / 3 / 4 / 5`
- 场景：`同构 / 异构 / 典型异构均值`
- `ISL = 5 Gbps`
- `GSL = 100 Mbps`
- `LEO` 同构算力：`3 TFLOPS`
- `LEO` 内存：`4 GB`
- `GS` 算力：`300 TFLOPS`
- `GS` 内存：`64 GB`

### 现象

- `YOLOv5`：同构场景变化很小，异构场景下可能略有变差
- `ResNet101`：异构场景下随节点数增加有收益
- `VGG19`：节点数增加时收益更明显
- `ViT-Huge`：基本保持 `GS-Only = 1`

### 结论

- 节点数增加本身不保证收益，效果与模型特征和资源异构性有关。
- `VGG19` 更能利用多星 `PMP` 的切分空间。
- `YOLOv5` 在某些异构条件下更容易受到额外通信开销影响。

### 复现命令

```powershell
python thesis_entry.py exp02 --repeats 1 --show-only
```

## 实验 3：LAWA-CDP 数据量敏感性

### 图文件

- `03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_resnet101.png`

### 参数

- 模型：`ResNet101`
- 输入数据量：`16 / 32 / 64 / 128`
- worker 数量：固定 `4`
- 场景：`同构 worker / 异构 worker`
- 算法：`LAWA / Greedy / Uniform / Random / Sat-Only`
- profile：`config/dnn_profiles_database_jetson.json`
- 随机基线：固定种子，重复 `30` 次

### 现象

- 同构场景下，`LAWA` 与简单分配差距较小
- 异构场景下，`LAWA` 明显优于 `Uniform / Random`
- 随输入数据量增大，`LAWA` 对异构 worker 的利用更稳定

### 结论

- `CDP` 的收益不仅来自并行，还来自对异构 worker 的感知分配。
- `LAWA` 在异构场景下优于简单均分与随机分配。

### 复现命令

```powershell
python thesis_entry.py exp03 --models resnet101 --show-only
```

## 实验 4：LAWA-CDP worker 数量敏感性

### 图文件

- `04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_yolov5.png`

### 参数

- 模型：`YOLOv5`
- 输入数据量：固定 `64`
- worker 数量：`1 / 2 / 3 / 4 / 5`
- 算法：`LAWA / Greedy / Sat-Only`
- profile：`config/dnn_profiles_database_jetson.json`
- 场景：异构 worker 集合逐步增加

### 现象

- `LAWA` 随 worker 数增加持续下降
- `Greedy` 也下降，但整体高于 `LAWA`
- `Sat-Only` 作为基线保持 `1`

### 结论

- 在当前异构 worker 配置下，增加 worker 数量能降低 `CDP` 时延。
- `LAWA` 比 `Greedy` 更能利用异构 worker。

### 复现命令

```powershell
python thesis_entry.py exp04 --models yolov5 --show-only
```

## 实验 5：FWMS 模式选择有效性

### 图文件

- `05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness.png`

### 参数

- 模型：`YOLOv5 / ResNet101 / VGG19 / ViT-Huge`
- 输入数据量：固定 `64`
- 模式：`PMP / CDP / GS-Only / Sat-Only / FWMS`
- `CDP worker_count = 4`
- `worker_memory = 2048 MB`
- `gsl_bandwidth = 100 Mbps`
- `gs_compute_factor = 100`
- 共同时间片集合：`28` 个 common slots
- 公平口径：
  - `PMP` 共享路由至少包含 `3` 颗 `LEO`
  - `CDP active_sat_count >= 3`
  - `PMP / GS-Only / Sat-Only` 在每个 slot 上共用同一路由

### 现象

- `YOLOv5`：`CDP` 最低，`FWMS` 接近 `CDP`
- `ResNet101`：`CDP` 最低，`FWMS` 接近 `CDP`
- `VGG19`：`CDP / Sat-Only` 不可行，`FWMS` 回到 `PMP`
- `ViT-Huge`：`CDP / Sat-Only` 不可行，`FWMS` 与 `PMP / GS-Only` 重合

### 结论

- 不同模型对应的最优模式不同。
- `FWMS` 会随模型特征与资源可行性切换模式，而不是固定选择某一种模式。

### 复现命令

```powershell
python thesis_entry.py exp05 --show-only
```

## 实验 6：FWMS 输入数据量敏感性

### 图文件

- `06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5.png`
- `06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5_normalized.png`

### 参数

- 模型：`YOLOv5`
- 输入数据量：`16 / 32 / 64 / 128`
- 模式：`PMP / CDP / GS-Only / Sat-Only / FWMS`
- `CDP worker_count = 4`
- `worker_memory = 2048 MB`
- `gsl_bandwidth = 100 Mbps`
- `gs_compute_factor = 100`
- 共同时间片集合：与实验 5 相同的 `28` 个 common slots
- 路由公平口径：与实验 5 相同

### 现象

- `GS-Only` 随 batch 增大近似线性增长
- `PMP` 保持稳定低于 `GS-Only`
- `CDP` 在 `16 / 32 / 64` 上最低，到 `128` 不可行
- `FWMS` 在小中 batch 下接近 `CDP`，到 `128` 回退为 `PMP`

### 结论

- 模式优劣会随任务规模变化。
- `FWMS` 在可行时选择更低时延模式，在不可行时回退到保底模式。

### 复现命令

```powershell
python thesis_entry.py exp06 --show-only
```
