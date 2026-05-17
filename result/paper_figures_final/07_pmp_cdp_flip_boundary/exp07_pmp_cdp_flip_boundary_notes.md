# 实验 7：PMP/CDP 翻转边界验证

- 模型：YOLOv5, ResNet101, VGG19, ViT-Huge。
- batch 固定为 16。
- CDP 并行卫星数：2, 3, 4。
- 每个模型-并行卫星数组合都直接运行 mode_selection_experiment.py，读取相同 STK 时隙集合的 PMP/CDP 结果。
- 比较类别定义：PMP更快 / CDP更快 / 仅PMP可行 / 仅CDP可行 / 均不可行 / 并列。
- 比值图中的 CDP/PMP > 1 表示 PMP 更快，< 1 表示 CDP 更快。