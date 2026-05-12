# 实验 00：模型层级输出特征图数据量分布

- profile：`config/dnn_profiles_database_pc.json`
- reference profile：`config/dnn_profiles_database_jetson.json`
- PC/Jetson 特征图数据量一致性检查：通过
- 展示批次：batch=1；若 profile 未包含该批次，则按最小可用 batch 线性折算。
- 绿色柱表示输入 input，绿色虚线表示输入大小。
- 红色柱表示该层输出大于 input，蓝色柱表示该层输出不大于 input。
- 最后一层输出标注为 result。

## 使用的源 profile key

- YOLOv5：`b16_640x640`
- ResNet101：`b16_224x224`
- VGG19：`b16_224x224`
- ViT-Huge：`b16_224x224`