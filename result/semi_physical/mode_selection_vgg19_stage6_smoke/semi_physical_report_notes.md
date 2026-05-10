# 半实物模式选择验证说明

本实验入口使用 PC/Jetson 真实推理与真实传输测量，再通过算力异构因子、链路带宽缩放和传播时延映射到卫星场景。

## 输入

- 理论结果表：`result\mode_selection\mode_selection_vgg19_stage6_feature_oracle_b64\data\slot_mode_results.csv`
- 网络配置：`config\network_config.json`
- 重复次数：`1`
- 单次最大真实传输负载：`8.0` MB

## 输出

- `semi_physical_mode_results.csv`：逐 slot、逐模式半实物结果。
- `semi_physical_summary.csv`：按模型、batch、模式汇总。
- `semi_physical_avg_latency_by_mode.png`：半实物平均时延对比。
- `semi_physical_theory_vs_real_latency.png`：理论与半实物趋势对比。

## 论文表述边界

该实验不是把真实卫星链路完全复现，而是在实验室设备上复现真实推理和真实网络传输，再映射到 STK 动态拓扑给出的异构资源条件。