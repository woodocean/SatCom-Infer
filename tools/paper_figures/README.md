# 论文绘图脚本索引

这个目录用于收口存放“论文图相关”的专题脚本，避免仓库根目录继续堆积一次性绘图入口。

当前脚本分工：

- `plot_paper_ready_figures.py`
  作用：根据已有结果重画论文主图。
  位置：保留在仓库根目录，作为长期主入口。

- `plot_runs_sensitivity_figures.py`
  作用：重画带宽敏感性、节点数量敏感性等折线图。
  位置：保留在仓库根目录，作为长期主入口。

- `tools/paper_figures/run_controlled_pmp_fig01.py`
  作用：重跑受控版 PMP 图 01。

- `tools/paper_figures/run_stk_slot_pmp_highlight.py`
  作用：针对单个 STK 时间片重跑 PMP，并对 `GA / Random` 做多次重复取均值。

- `tools/paper_figures/plot_stk_slot_highlight.py`
  作用：直接基于已有 `results_long_stk_dynamic.csv` 抽取某个时间片并画单时间片图，不重跑实验。

推荐使用原则：

1. 如果只是“根据已有 csv 重画”，优先用根目录的长期入口。
2. 如果是“针对某个时间片或某个受控场景专题重跑”，用这个目录下的专题脚本。
3. 后续新增论文图脚本，优先放到这个目录，不再直接丢到仓库根目录。
