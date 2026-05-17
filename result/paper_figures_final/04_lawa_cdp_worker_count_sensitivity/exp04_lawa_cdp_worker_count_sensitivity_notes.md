# 实验 4：LAWA-CDP 模式的 worker 数量敏感性实验

## 模型

YOLOv5

## 验收重跑命令

```powershell
python thesis_entry.py exp04 --out-dir result/paper_figures_final/04_lawa_cdp_worker_count_sensitivity
```

## 结论口径

- 实验 4 使用异构 worker 场景、batch=64，并将任务总量固定为 50 个任务块。
- 图中仅保留 LAWA-CDP 与 Single-LEO 两种方式，比较分钟级总时延。
- 随着参与 LEO 节点数增加，LAWA 更能把新增节点资源转化为总时延收益。