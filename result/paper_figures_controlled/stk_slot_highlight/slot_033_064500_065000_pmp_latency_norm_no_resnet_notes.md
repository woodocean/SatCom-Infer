# slot_033_064500_065000 单时间片图

## Shared slot parameters

- ISL 平均带宽：`9690.3498 Mbps`
- GSL 平均带宽：`116.9512 Mbps`
- 流水线节点数：`5`
- 跳数：`6`
- 路径：`RS->SAT-01->SAT-02->SAT-03->SAT-04->SAT-05->GS`

## Model settings

| 模型 | batch | 输入尺寸 | LADP归一化时延 |
|---|---:|---:|---:|
| YOLOv5 | 32 | 640x640 | 0.352 |
| VGG19 | 32 | 224x224 | 0.207 |
| Swin-Base | 32 | 224x224 | 0.750 |
| ViT-Huge | 32 | 224x224 | 1.000 |