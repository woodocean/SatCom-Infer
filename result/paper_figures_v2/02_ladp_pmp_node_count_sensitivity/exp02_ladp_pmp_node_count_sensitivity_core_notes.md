# PMP 模式节点数量敏感性分析

- 图类型：折线图
- 横坐标：中继 LEO 卫星数量
- 纵坐标：归一化时延，GS-Only = 1
- 模型：YOLOv5, VGG19
- 算法：LADP, 贪心, 遗传算法
- 数据来源：
  - `result\runs\20260510_174011_exp02_nodes_yolov5_20260510_174008_node_count_sensitivity_yolov5_b32_640x640_theory_values_p5_r30_seed42\data\results_long_node_count_sensitivity_yolov5_b32_640x640_theory_values_p5_r30_seed42.csv`
  - `result\runs\20260510_174021_exp02_nodes_vgg19_20260510_174008_node_count_sensitivity_vgg19_b32_224x224_theory_values_p5_r30_seed42\data\results_long_node_count_sensitivity_vgg19_b32_224x224_theory_values_p5_r30_seed42.csv`

- 额外趋势图：`exp02_ladp_pmp_node_count_sensitivity_core_la_dp_trend.png/pdf`
- 趋势图算法：`LADP`