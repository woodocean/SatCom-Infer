# 实验 5：FWMS 模式选择有效性实验

- 本实验直接读取 mode_selection_experiment.py 生成的 slot_mode_results.csv。
- 共同时间片集合：所有模型共享同一批 slot，并要求 PMP/GS-Only/FWMS 可行且 PMP 路由至少包含 3 颗 LEO。
- 统计口径：各模式在共同 slot 集合上取均值；CDP 仅统计 active_sat_count 至少为 3 的可行 slot。
- 路由口径：PMP、GS-Only、Sat-Only 在每个 slot 上共用同一路由；Sat-Only 只在该共享路由上选择单星执行位置。