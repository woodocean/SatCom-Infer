# paper_figures_v3

本目录用于保存基于当前新模型 profile 数据重跑后的论文实验 1-6 图和汇总数据。

## 数据口径

- PMP/切分实验读取当前 `config/dnn_profiles_database_pc.json` 与 `config/dnn_profiles_database_jetson.json`。
- CDP/FWMS 理论实验读取当前 `config/dnn_profiles_database_jetson.json`。
- 除模型 profile 数据更新外，其余参数沿用 `paper_figures_v2` 的实验 1-6 口径。
- 实验 5 不补半实物仿真数据，只保留理论仿真部分和半实物占位说明。

## 一键重跑命令

实验 1：

```powershell
python thesis_entry.py exp01 -- `
  --slot-id slot_033_064500_065000 `
  --models yolov5,resnet101,vgg19,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v3\01_ladp_pmp_algorithm_effectiveness
```

实验 2：

```powershell
python thesis_entry.py exp02 `
  --out-dir result\paper_figures_v3\02_ladp_pmp_node_count_sensitivity
```

实验 3：

```powershell
python thesis_entry.py exp03 `
  --out-dir result\paper_figures_v3\03_lawa_cdp_data_sensitivity
```

实验 4：

```powershell
python thesis_entry.py exp04 `
  --out-dir result\paper_figures_v3\04_lawa_cdp_worker_count_sensitivity
```

实验 5：

```powershell
python thesis_entry.py exp05 `
  --min-pmp-route-leo 3 `
  --min-cdp-active-sats 3 `
  --out-dir result\paper_figures_v3\05_fwms_mode_selection_effectiveness
```

实验 6：

```powershell
python thesis_entry.py exp06 `
  --min-pmp-route-leo 3 `
  --min-cdp-active-sats 3 `
  --out-dir result\paper_figures_v3\06_fwms_data_sensitivity
```

实验 5/6 公平口径说明：

- 不再使用实验 1 的 `pmp_summary` 归一化外推 PMP。
- 实验 5/6 读取 `mode_selection_experiment.py` 生成的 `slot_mode_results.csv`。
- 筛选代表性 slot：PMP route 至少 3 颗 LEO，CDP 可比时 active_sat_count 至少 3。
- 对有 CDP 可比性的模型或 batch，优先使用共同代表 slot；无 CDP 可比 slot 的项会在 `slot_filter.csv` 中标记为 fallback。

## 已生成产物

### 实验 1：LADP-PMP 算法有效性

- [exp01_ladp_pmp_algorithm_effectiveness.png](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.png)
- [exp01_ladp_pmp_algorithm_effectiveness.pdf](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness.pdf)
- [exp01_ladp_pmp_algorithm_effectiveness_summary.csv](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_summary.csv)
- [exp01_ladp_pmp_algorithm_effectiveness_long.csv](./01_ladp_pmp_algorithm_effectiveness/exp01_ladp_pmp_algorithm_effectiveness_long.csv)

### 实验 2：LADP-PMP 节点数量敏感性

- [exp02_ladp_pmp_node_count_sensitivity.png](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity.png)
- [exp02_ladp_pmp_node_count_sensitivity.pdf](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity.pdf)
- [exp02_ladp_pmp_node_count_sensitivity_active_sat_count.png](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_active_sat_count.png)
- [exp02_ladp_pmp_node_count_sensitivity_active_sat_count.pdf](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_active_sat_count.pdf)
- [exp02_ladp_pmp_node_count_sensitivity_summary.csv](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_summary.csv)
- [exp02_ladp_pmp_node_count_sensitivity_long.csv](./02_ladp_pmp_node_count_sensitivity/exp02_ladp_pmp_node_count_sensitivity_long.csv)

### 实验 3：LAWA-CDP 数据量敏感性

- [exp03_lawa_cdp_data_sensitivity_yolov5.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_yolov5.png)
- [exp03_lawa_cdp_data_sensitivity_resnet101.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_resnet101.png)
- [exp03_lawa_cdp_data_sensitivity_vgg19.png](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_vgg19.png)
- [exp03_lawa_cdp_data_sensitivity_summary.csv](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_summary.csv)
- [exp03_lawa_cdp_data_sensitivity_long.csv](./03_lawa_cdp_data_sensitivity/exp03_lawa_cdp_data_sensitivity_long.csv)

### 实验 4：LAWA-CDP worker 数量敏感性

- [exp04_lawa_cdp_worker_count_sensitivity_yolov5.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_yolov5.png)
- [exp04_lawa_cdp_worker_count_sensitivity_resnet101.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_resnet101.png)
- [exp04_lawa_cdp_worker_count_sensitivity_vgg19.png](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_vgg19.png)
- [exp04_lawa_cdp_worker_count_sensitivity_summary.csv](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_summary.csv)
- [exp04_lawa_cdp_worker_count_sensitivity_long.csv](./04_lawa_cdp_worker_count_sensitivity/exp04_lawa_cdp_worker_count_sensitivity_long.csv)

### 实验 5：FWMS 模式选择有效性

- [exp05_fwms_mode_selection_effectiveness.png](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness.png)
- [exp05_fwms_mode_selection_effectiveness.pdf](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness.pdf)
- [exp05_fwms_mode_selection_effectiveness_summary.csv](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness_summary.csv)
- [exp05_fwms_mode_selection_effectiveness_slot_filter.csv](./05_fwms_mode_selection_effectiveness/exp05_fwms_mode_selection_effectiveness_slot_filter.csv)

### 实验 6：FWMS 输入数据量敏感性

- [exp06_fwms_data_sensitivity_yolov5.png](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5.png)
- [exp06_fwms_data_sensitivity_yolov5.pdf](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5.pdf)
- [exp06_fwms_data_sensitivity_yolov5_summary.csv](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5_summary.csv)
- [exp06_fwms_data_sensitivity_yolov5_slot_filter.csv](./06_fwms_data_sensitivity/exp06_fwms_data_sensitivity_yolov5_slot_filter.csv)
