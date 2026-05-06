# 半实物模式选择验证入口说明

入口脚本：

```bash
python semi_physical_mode_verify.py
```

它的定位不是替代理论仿真，而是复刻几个关键理论实验：在 PC / Jetson 上真实推理、真实传输，然后用算力异构因子、带宽缩放和传播时延映射到 STK 动态拓扑场景。

## 1. 设备配置

先生成设备模板：

```bash
python semi_physical_mode_verify.py write-template --network-config config\network_config.json --output config\semi_physical_devices.example.json
```

然后复制一份作为实际配置：

```bash
copy config\semi_physical_devices.example.json config\semi_physical_devices.local.json
```

需要手动修改 `Jetson_1`、`Jetson_2` 的 `repo` 字段，改成 Jetson 上的项目路径，例如：

```json
"repo": "/home/nvidia/Neurosurgeon-main"
```

## 2. 单节点推理测量

在 PC 或 Jetson 本机测试真实推理：

```bash
python semi_physical_mode_verify.py measure-infer --model-name yolov5 --batch-size 32 --input-h 640 --input-w 640 --start-layer 0 --end-layer -1 --repeats 3
```

脚本会输出一行 `SEMI_JSON:{...}`，里面包含真实推理均值、方差和设备类型。

## 3. 复刻关键理论实验

推荐先从 YOLO 的 Stage6 结果开始：

```bash
python semi_physical_mode_verify.py run ^
  --mode-results-csv result\mode_selection\mode_selection_yolo_stage6_feature_oracle_b64\data\slot_mode_results.csv ^
  --network-config config\network_config.json ^
  --device-config config\semi_physical_devices.local.json ^
  --output-dir result\semi_physical\semi_physical_yolo_b64_stage6 ^
  --modes PMP,CDP,GS-Only,FWMS-Feature ^
  --limit-slots 2 ^
  --repeats 3 ^
  --max-transfer-mb 32
```

输出：

- `semi_physical_mode_results.csv`：逐时间片、逐模式结果。
- `semi_physical_summary.csv`：按模式汇总的理论/半实物平均时延。
- `semi_physical_avg_latency_by_mode.png`：半实物模式平均时延图。
- `semi_physical_theory_vs_real_latency.png`：理论与半实物趋势对比图。
- `semi_physical_report_notes.md`：实验说明。

## 4. 半实物仿真逻辑

计算部分：

- PC / Jetson 真实加载模型并执行推理。
- 对逻辑卫星算力使用缩放：`半实物等效计算时延 = 真实推理时延 * 物理设备算力 / 逻辑节点算力`。
- 这样可以用两台 Jetson 模拟多颗异构卫星。

通信部分：

- PC 与 Jetson 之间用真实文件传输测量物理链路传输时间。
- 对目标卫星链路带宽使用缩放：`等效传输时延 = 真实传输时延 * 物理链路基准带宽 / 目标链路带宽`。
- 再叠加 `network_config` 或模式配置中的传播时延。

模式复刻：

- `PMP`：按 LADP 的层切分计划测量各段推理，并按流水线链路累计通信。
- `CDP`：按 LAWA 的 batch 分配测量各 worker 全模型推理，端到端时延取最慢 worker 分支。
- `GS-Only`：测量输入回传和 GS 全模型推理。
- `FWMS-Feature`：读取理论实验中的选择结果，再取对应基础模式的半实物结果。

## 5. 论文表述边界

建议表述为：

> 本文基于 PC 与两台 Jetson Orin NX 搭建半实物仿真平台，由真实设备执行模型推理并测量物理网络传输开销，再结合 STK 动态拓扑给出的可见性、链路带宽和传播时延参数，对分布式卫星协作推理模式进行等效验证。

不要表述为“完全复现真实卫星网络”。更稳的说法是“半实物平台验证理论趋势和关键开销模型的合理性”。
