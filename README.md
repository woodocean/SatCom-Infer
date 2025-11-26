# SatCom-Infer：面向分布式卫星协同推理的高效通信策略

一个分布式卫星协同推理系统，支持在具有异构计算能力的多卫星节点间进行高效的DNN模型拆分与协同推理。

## 项目特性

- 多卫星协同推理：将DNN模型层分布到卫星网络中执行
- 智能模型分割：基于计算资源和网络条件的最优层分配策略
- 时延预测：预训练模型用于准确的计算和传输时延估计
- 异构算力支持：适配不同计算能力的卫星节点
- 地面站集成：支持将最终结果传输到地面站

## 项目结构

SatCom-Infer/
├── satellite_node.py          # 卫星节点核心实现
├── satellite_api.py           # 卫星节点API服务
├── ground_station_api.py      # 地面站API服务
├── task_client.py             # 任务提交客户端
├── models/                    # DNN模型定义
│   ├── AlexNet.py
│   ├── LeNet.py
│   ├── MobileNet.py
│   └── VggNet.py
├── utils/                     # 工具模块
│   ├── inference_utils.py     # 推理工具函数
│   └── excel_utils.py
├── predictor/                 # 时延预测器
│   ├── predictor_utils.py
│   └── config/               # 预测模型配置
└── net/                      # 网络通信模块
    └── net_utils.py

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- 其他依赖见 requirements.txt

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/your-username/SatCom-Infer.git
cd SatCom-Infer
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

### 运行演示

1. 启动地面站（终端1）
```bash
python ground_station_api.py --station_id GROUND-001 --ip 127.0.0.1 --port 20001
```

2. 启动卫星节点（终端2、3、4）
```bash
# 遥感卫星
python satellite_api.py --node_id SAT-001 --satellite_type remote_sensing --ip 127.0.0.1 --port 10001 --compute_capacity 8.0 --device cuda --ground_station_id GROUND-001 --ground_station_ip 127.0.0.1 --ground_station_port 20001

# 计算卫星1
python satellite_api.py --node_id SAT-002 --satellite_type leo_computing --ip 127.0.0.1 --port 10002 --compute_capacity 6.0 --device cuda

# 计算卫星2
python satellite_api.py --node_id SAT-003 --satellite_type leo_computing --ip 127.0.0.1 --port 10003 --compute_capacity 4.0 --device cpu
```

3. 提交推理任务（终端5）
```bash
python task_client.py
```

## 核心算法

### 模型分割策略
系统根据卫星节点的计算能力、网络带宽和时延约束，智能地将DNN模型分割成多个部分，分配到不同的卫星节点执行。

### 时延预测
使用预训练的线性回归模型预测各类型DNN层在不同设备上的推理时延，为分割决策提供依据。

### 协同调度
遥感卫星作为协调节点，收集网络状态信息，计算最优分割方案，并协调各节点完成协同推理。

## 引用声明

本项目基于 [Neurosurgeon](https://github.com/Tjyy-1223/Neurosurgeon) 进行开发，特此感谢原作者的贡献。

### 主要借鉴内容
- 模型拆分框架设计
- 时延预测模型实现
- DNN层时延分析工具
- 模型分区基础算法

### 我们的改进与扩展
- 添加了卫星网络仿真环境
- 实现了分布式卫星节点发现与通信
- 开发了多卫星协同推理调度算法
- 增加了地面站集成支持
- 优化了异构算力资源管理

## 许可证

本项目基于MIT许可证发布，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues提交。

