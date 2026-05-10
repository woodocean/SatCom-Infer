# Controlled PMP Fig. 01

## Experiment setup

- Pipeline: `RS -> SAT-01 -> SAT-02 -> SAT-03 -> GS`.
- ISL bandwidth: `1800 Mbps`; GSL bandwidth: `100 Mbps`.
- LEO compute: `5.0 TFLOPS`; GS compute: `500.0 TFLOPS`.
- LEO memory: `4096 MB`.
- Propagation delays are inherited from the selected STK path.
- Repeats: `100` for each model, mainly to stabilize Random and GA.
- Energy model: `P_compute=15 W`, `P_tx=10 W`.

## Input profiles

| Model | Batch | Input |
|---|---:|---:|
| YOLOv5 | 64 | 640x640 |
| VGG19 | 64 | 224x224 |
| Swin-Base | 64 | 224x224 |
| ViT-Huge | 64 | 224x224 |

## Best normalized latency

| Model | Best algorithm | Mean normalized latency |
|---|---|---:|
| YOLOv5 | LA-DP | 0.373 |
| VGG19 | LA-DP | 0.643 |
| Swin-Base | LA-DP | 0.990 |
| ViT-Huge | LA-DP | 1.000 |

## Files

- Config: `result/paper_figures_controlled/fig01_pmp_algorithm/controlled_three_leo_pmp_config.json`
- Long results: `result/paper_figures_controlled/fig01_pmp_algorithm/fig01_pmp_controlled_results_long.csv`
- Summary: `result/paper_figures_controlled/fig01_pmp_algorithm/fig01_pmp_controlled_summary.csv`
- Figure: `01_pmp_algorithm_latency_norm_controlled.png/pdf`