# 实验 3：LAWA-CDP 模式的数据量敏感性实验

## 模型

ResNet101

## 验收重跑命令

```powershell
python thesis_entry.py exp03 --out-dir result/paper_figures_final/03_lawa_cdp_data_sensitivity
```

## 结论口径

- 实验 3 使用 PC profile，并仅保留异构 worker 场景。
- 横轴 batch 从 64 扩展到 512，用于观察 CDP 在更大任务输入下的相对时延变化。
- LAWA 的优势主要体现在异构节点间的数据分配能力。