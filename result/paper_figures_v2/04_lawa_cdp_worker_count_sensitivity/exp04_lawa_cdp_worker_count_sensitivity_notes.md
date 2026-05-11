# 实验 4：LAWA-CDP 模式的 worker 数量敏感性实验

## 模型

YOLOv5, ResNet101, VGG19

## 验收重跑命令

```powershell
python thesis_entry.py exp04 --out-dir result\paper_figures_v2\04_lawa_cdp_worker_count_sensitivity
```

## 结论口径

- worker 数量增加通常会降低 CDP 时延，但收益不是线性的。
- 当新增 worker 算力或链路条件较弱时，额外分发和回传开销会削弱并行收益。
- LAWA 的作用是根据 worker 的计算和链路状态调节数据量，避免简单均分在异构场景下失效。