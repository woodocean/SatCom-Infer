# 验收结构与统一命令

这份文档只服务于结题验收和最终汇报，避免被历史脚本、历史结果和旧说明干扰。

## 1. 当前建议只关注的根目录入口

- `thesis_entry.py`
  - 统一入口。后续验收、重跑图、调用不同实验都优先从这里进。
- `experiments_runner.py`
  - 传统 PMP 理论实验主入口。
- `stk_dynamic_experiment.py`
  - STK 动态拓扑 PMP 实验主入口。
- `mode_selection_experiment.py`
  - 模式选择实验主入口。
- `main.py`
  - 单节点运行入口，服务于物理/半实物节点。

## 2. 已归类到 tools 的工具脚本

下面这些不再建议直接在根目录维护，统一视为工具脚本：

- `tools/build_stk_network_config.py`
- `tools/plot_avg_tho_vs_real.py`
- `tools/plot_mode_selection_summary.py`
- `tools/plot_paper_ready_figures.py`
- `tools/plot_runs_sensitivity_figures.py`
- `tools/plot_stk_cross_model_summary.py`
- `tools/semi_physical_mode_verify.py`
- `tools/physical_experiment_orchestrator.py`
- `tools/paper_figures/run_stk_slot_pmp_highlight.py`

## 3. 统一入口怎么用

### 实验主线

```powershell
python thesis_entry.py run-legacy ...
python thesis_entry.py run-stk ...
python thesis_entry.py run-mode ...
```

### 前处理

```powershell
python thesis_entry.py build-stk-config ...
```

### 绘图

```powershell
python thesis_entry.py plot-legacy ...
python thesis_entry.py plot-mode-summary ...
python thesis_entry.py plot-paper ...
python thesis_entry.py plot-sensitivity ...
python thesis_entry.py plot-stk-summary ...
```

### 半实物

```powershell
python thesis_entry.py semi-physical ...
python thesis_entry.py physical-orchestrator ...
```

### 论文实验 1 重跑

```powershell
python thesis_entry.py exp01 `
  --slot-id slot_033_064500_065000 `
  --models yolov5,vgg19,swin_base,vit_huge `
  --repeats 100 `
  --out-dir result\paper_figures_v2\01_ladp_pmp_algorithm_effectiveness
```

## 4. 验收时只建议打开的结果目录

- `result/paper_figures_v2/`
  - 最终图、参数口径、重跑步骤统一入口
- `result/stk_dynamic/`
  - STK 动态拓扑理论实验原始结果
- `result/mode_selection/`
  - 模式选择实验原始结果

## 5. 验收时不建议展开讲的目录

- `waste/`
- `.codex_tmp/`
- `result/paper_figures/`
- `result/paper_figures_controlled/`
- `result/v1.0/`, `result/v2.0/`, `result/v3.0/`, `result/v4.0/`
- `runs/`

这些可以保留，但不作为汇报主线。

## 6. 验收推荐顺序

1. 打开 `result/paper_figures_v2/README.md`
2. 先讲实验 1 的图、参数和结论
3. 如老师要求复现，现场运行 `python thesis_entry.py exp01 ...`
4. 再讲后续实验会继续按 `paper_figures_v2` 的统一结构补齐

## 7. 现在这次整理的目标

这次不是彻底重构所有历史代码，而是：

- 把工具脚本从根目录收口到 `tools/`
- 在根目录新增一个统一入口 `thesis_entry.py`
- 把验收命令和最终图入口统一到 `result/paper_figures_v2/` 和本说明文档

这样做的目的是降低验收时的混乱度，不影响已有实验结果。
