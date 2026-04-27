# STK Cross-Model Summary

Lower is better for both normalized latency and satellite energy.

## Average Normalized Latency (GS-Only=1.0)

| Model | LA-DP | Greedy | Random | GA | Uniform | GS-Only |
|---|---|---|---|---|---|---|
| YOLOv5 | **0.368** | 0.372 | 0.619 | 0.413 | 0.712 | 1.000 |
| ResNet101 | **0.859** | 0.943 | 1.562 | 0.922 | 1.570 | 1.000 |
| VGG19 | **0.342** | 0.586 | 2.740 | 0.459 | 1.517 | 1.000 |
| Swin-Base | **0.906** | 0.957 | 1.832 | 1.494 | 2.071 | 1.000 |
| ViT-Huge | **1.000** | **1.000** | 9.544 | 2.709 | 8.769 | **1.000** |

## Average Satellite Energy (J)

| Model | LA-DP | Greedy | Random | GA | Uniform | GS-Only |
|---|---|---|---|---|---|---|
| YOLOv5 | **31.486** | 31.894 | 55.008 | 38.317 | 64.787 | 89.412 |
| ResNet101 | **9.715** | 10.496 | 18.148 | 10.854 | 18.299 | 10.953 |
| VGG19 | **3.812** | 7.042 | 35.277 | 5.520 | 17.467 | 10.953 |
| Swin-Base | **10.345** | 10.772 | 23.816 | 22.162 | 27.179 | 10.953 |
| ViT-Huge | **10.953** | **10.953** | 135.633 | 56.878 | 128.074 | **10.953** |

## Best Algorithm Per Model

| Model | Best Latency Algo | Best Latency | Best Energy Algo | Best Energy (J) |
|---|---|---:|---|---:|
| YOLOv5 | LA-DP | 0.368 | LA-DP | 31.486 |
| ResNet101 | LA-DP | 0.859 | LA-DP | 9.715 |
| VGG19 | LA-DP | 0.342 | LA-DP | 3.812 |
| Swin-Base | LA-DP | 0.906 | LA-DP | 10.345 |
| ViT-Huge | LA-DP | 1.000 | LA-DP | 10.953 |
