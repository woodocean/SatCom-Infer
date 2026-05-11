# 实验 3：LAWA-CDP 模式的数据量敏感性实验

## 模型

YOLOv5, ResNet101, VGG19

## 验收重跑命令

```powershell
python thesis_entry.py exp03 --out-dir result\paper_figures_v2\03_lawa_cdp_data_sensitivity
```

## 结论口径

- 同构 worker 场景下，各 worker 能力一致，均匀分配通常接近 LAWA。
- 异构 worker 场景下，LAWA 会把更多数据分给算力强、链路好的 worker，优势更明显。
- 输入数据量增大后，离散样本分配更接近连续最优解，LAWA 相对随机/贪心/均匀的稳定性更容易体现。