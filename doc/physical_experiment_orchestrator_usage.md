# PC + Jetson 半实物一键编排说明

入口脚本：

```bash
python physical_experiment_orchestrator.py
```

这个入口负责：

- 把当前 `network_config.json` 同步到两台 Jetson。
- 在 Jetson 的 Docker 容器中启动逻辑卫星节点。
- 在 PC 上启动需要的地面节点。
- 调用现有 `experiments_runner.py --exp-mode physical|hybrid` 下发任务。

## 1. 设备信息

默认设备配置：

```json
{
  "Jetson_1": {
    "host": "192.168.10.181",
    "user": "nvidia",
    "password": "nvidia",
    "repo": "/home/nvidia/satinfer/SatCom-Infer"
  },
  "Jetson_2": {
    "host": "192.168.10.178",
    "user": "nvidia",
    "password": "nvidia",
    "repo": "/home/nvidia/satinfer/SatCom-Infer"
  }
}
```

生成模板：

```bash
python physical_experiment_orchestrator.py write-template --output config\physical_jetsons.example.json
```

建议复制成 local 文件：

```bash
copy config\physical_jetsons.example.json config\physical_jetsons.local.json
```

## 2. 先做 dry-run

dry-run 不会连接 Jetson，只打印将要执行的命令：

```bash
python physical_experiment_orchestrator.py run-pmp ^
  --dry-run ^
  --source-config config\network_config.json ^
  --runtime-config config\network_config.json ^
  --jetson-config config\physical_jetsons.local.json ^
  --run-id dry_physical_demo ^
  --num-tasks 1
```

确认输出里包含：

- `sync config -> Jetson_1`
- `sync config -> Jetson_2`
- `launch SAT-xx`
- `local launch GS`
- `experiments_runner.py --exp-mode physical`

## 3. 跑 PMP 半实物实验

当前最稳的真实分布式链路是 PMP/LADP，因为现有 `ComputeNode` 已经支持 PMP 真实推理、真实 UDP 传输、通信时延测量和算力/带宽缩放。

```bash
python physical_experiment_orchestrator.py run-pmp ^
  --source-config config\network_config.json ^
  --runtime-config config\network_config.json ^
  --jetson-config config\physical_jetsons.local.json ^
  --run-id physical_pmp_yolo_b32_demo ^
  --num-tasks 3 ^
  --model-name yolov5 ^
  --batch-size 32 ^
  --input-h 640 ^
  --input-w 640
```

执行流程：

1. 远程连接 `192.168.10.181` 和 `192.168.10.178`。
2. 同步 `config/network_config.json` 到 `/home/nvidia/satinfer/SatCom-Infer/config/network_config.json`。
3. 用 Docker 启动逻辑卫星节点：

```bash
cid=$(docker run -d --rm --name <container> \
  --runtime nvidia --network host \
  -v /home/nvidia/satinfer/SatCom-Infer:/workspace \
  -w /workspace satinfer:v4.0 \
  bash -lc 'python main.py --id SAT-xx')
nohup docker logs -f $cid > logs/<run_id>_SAT-xx.log 2>&1 < /dev/null &
```

4. PC 本地启动 `GS`。
5. `experiments_runner.py` 在本地启动进程内 `RS`，下发 PMP 任务。
6. 任务结束后自动清理本地进程和远程 Docker 容器。

## 4. 为什么当前 run-pmp 不单独启动 RS

现有 `experiments_runner.py` 的物理模式不是通过 UDP 把任务发给外部 RS 进程，而是在 runner 进程内创建 `ComputeNode(RS)`，然后直接调用 `rs_node.handle_message(...)`。

所以 `run-pmp` 默认只额外启动：

- 远程 SAT 节点。
- 本机 GS 节点。

RS 由 runner 自己启动。这样最稳，不会出现两个 RS 抢同一个端口的问题。

## 5. 如何接 STK 动态拓扑

如果要用某个 STK 时间片配置，可以把 `--source-config` 指到对应 slot 的配置：

```bash
python physical_experiment_orchestrator.py run-pmp ^
  --source-config result\stk_dynamic\stk_dynamic_yolo_001\configs\slot_001_040500_041000_network_config.json ^
  --runtime-config config\network_config.json ^
  --jetson-config config\physical_jetsons.local.json ^
  --run-id physical_pmp_stk_slot001 ^
  --num-tasks 3
```

脚本会把该 slot 配置复制成运行时 `config/network_config.json`，再同步到 Jetson。

## 6. 后续扩展 CDP / FWMS

当前入口先完成 PMP 半实物闭环。下一步要做 CDP/FWMS，需要继续补两件事：

- 把 `ComputeNode` 的 CDP 路径改成论文里的无聚合器版：worker 全模型推理后直接回 GS。
- 把 CDP 的物理结果写入和 PMP 一样的标准长表，方便 FWMS 读取真实半实物结果。

在论文里可以先把当前版本表述为：

> 半实物平台首先验证 PMP/LADP 的真实分布式执行链路，并在相同设备和网络条件下复现实测计算、通信开销。CDP/FWMS 的半实物扩展采用同一编排框架，可进一步接入无聚合器数据并行链路。
