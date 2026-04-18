"""
Router QoS 集成示例与测试脚本

展示如何在项目中集成 RouterQoSClient：
1. 单条规则应用
2. 从配置文件批量应用
3. 在 main.py 中的使用点
4. 与仿真器的对齐
"""

import json
import os
import sys
from typing import Dict

# 确保能导入 router_qos
sys.path.insert(0, os.path.dirname(__file__))
from router_qos import RouterQoSClient, RouterQoSException


def example_single_rule(router_ip: str, username: str, password: str):
    """示例 1: 添加单条带宽+延迟规则"""
    print("=" * 60)
    print("示例 1: 添加单条规则")
    print("=" * 60)
    
    with RouterQoSClient(router_ip, username, password) as qos:
        # 添加一条 UDP 规则：PC 8000→Jetson 5000，限速 10Mbps，延迟 100ms
        result = qos.add_bandwidth_limit_with_delay(
            src_ip='192.168.10.165', dst_ip='192.168.3.181',
            src_port=8000, dst_port=5000,
            rate_mbit=10.0, delay_ms=100, jitter_ms=10,
            protocol='udp', verify=True
        )
        
        print(f"\n结果: {json.dumps(result, indent=2)}")
        
        # 列出所有规则
        print("\n当前规则列表:")
        for rule in qos.list_rules():
            print(f"  {rule['src_ip']}:{rule['src_port']} → {rule['dst_ip']}:{rule['dst_port']} | "
                  f"{rule['rate_mbit']} Mbps, {rule['delay_ms']}ms, mark={rule['mark']}")


def example_batch_from_config(router_ip: str, username: str, password: str, config_path: str):
    """示例 2: 从配置文件批量应用规则"""
    print("=" * 60)
    print("示例 2: 从配置文件批量应用规则")
    print("=" * 60)
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    with RouterQoSClient(router_ip, username, password) as qos:
        result = qos.apply_rules_from_config(config_path, verify=True, dry_run=False)
        
        print(f"\n批量应用结果:")
        print(f"  应用成功: {result['applied_count']}")
        print(f"  应用失败: {result['failed_count']}")
        print(f"  详细结果: {json.dumps(result['results'], indent=2)}")


def example_dry_run(router_ip: str, username: str, password: str):
    """示例 3: 干运行（检查但不执行）"""
    print("=" * 60)
    print("示例 3: 干运行模式")
    print("=" * 60)
    
    with RouterQoSClient(router_ip, username, password) as qos:
        result = qos.add_bandwidth_limit_with_delay(
            src_ip='192.168.10.165', dst_ip='192.168.3.181',
            src_port=9000, dst_port=5050,
            rate_mbit=66.0, delay_ms=50,
            protocol='udp', dry_run=True
        )
        
        print(f"\n干运行结果（不会实际修改路由器）:")
        print(f"  {json.dumps(result, indent=2)}")


def example_cleanup(router_ip: str, username: str, password: str):
    """示例 4: 清除所有规则"""
    print("=" * 60)
    print("示例 4: 清除所有规则")
    print("=" * 60)
    
    with RouterQoSClient(router_ip, username, password) as qos:
        # 先列出当前规则
        print("\n当前规则:")
        for rule in qos.list_rules():
            print(f"  mark={rule['mark']}, {rule['src_ip']}→{rule['dst_ip']}")
        
        # 清除所有规则
        result = qos.clear_all()
        print(f"\n清理结果: {json.dumps(result, indent=2)}")


def integration_with_main_scheduler(router_ip: str, username: str, password: str, 
                                    net_config_path: str = "config/network_config.json"):
    """
    示例 5: 在 main.py/scheduler 中的集成用法
    
    这展示了如何在动态网络拓扑更新中融入真实路由器控制
    """
    print("=" * 60)
    print("示例 5: 与 main.py/scheduler 集成")
    print("=" * 60)
    
    # 模拟 main.py 中的用法：在每个任务周期更新网络拓扑
    def update_network_with_router_control(net_config_path: str):
        """
        替换原有的 main.update_network_topology 函数
        
        原逻辑：只修改 JSON 配置中的 bandwidth_mbps
        新逻辑：还要同时下发到真实路由器
        """
        
        # 1. 读取网络配置
        with open(net_config_path, 'r') as f:
            config = json.load(f)
        
        # 2. 修改 JSON 中的带宽（动态仿真）
        import random
        for link_key, link_info in config.get('links', {}).items():
            original_bw = link_info.get('bandwidth_mbps', 100)
            # 动态波动 ±20%
            new_bw = original_bw * (0.8 + random.random() * 0.4)
            link_info['bandwidth_mbps'] = new_bw
        
        # 3. 保存到 JSON
        with open(net_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 4. 同时下发到真实路由器
        print(f"\n正在同步规则到路由器 {router_ip}...")
        try:
            with RouterQoSClient(router_ip, username, password) as qos:
                result = qos.apply_rules_from_config(net_config_path, verify=True, dry_run=False)
                print(f"✓ 路由器规则已更新: {result['applied_count']} 条成功，{result['failed_count']} 条失败")
        except RouterQoSException as e:
            print(f"⚠ 路由器控制失败（仿真继续）: {e}")
    
    # 调用该函数
    if os.path.exists(net_config_path):
        update_network_with_router_control(net_config_path)
    else:
        print(f"配置文件不存在: {net_config_path}（跳过真实路由器控制，仅模拟）")


# ============================================================================
# 在 main.py 中的使用建议
# ============================================================================

"""
【在 main.py 中的集成示例】

1. 导入模块：
   from core.router_qos import RouterQoSClient, RouterQoSException

2. 初始化（在 main() 函数中）：
   qos_client = RouterQoSClient(
       host='192.168.10.1',
       username='root',
       password='wslhy110'
   )

3. 替换原有的 update_network_topology()：
   
   def update_network_topology_v2(config_path, qos_client):
       '''更新网络拓扑并同步到路由器'''
       # 原有逻辑：修改 bandwidth_mbps
       ...
       
       # 新增逻辑：同步到路由器
       try:
           result = qos_client.apply_rules_from_config(config_path, verify=True, dry_run=False)
           if result['ok']:
               print(f"✓ 路由器规则已应用: {result['applied_count']} 条")
           else:
               print(f"⚠ 部分规则应用失败，仿真继续")
       except Exception as e:
           print(f"⚠ 路由器控制异常: {e}（仿真继续）")

4. 在任务发起循环中调用：
   
   if args.id == "RS":
       time.sleep(3)
       from core.scheduler import Scheduler
       
       for i in range(61, 81):
           # ... 现有逻辑 ...
           
           # 更新网络拓扑（含路由器同步）
           update_network_topology_v2(net_config_path, qos_client)
           
           # 调度器感知更新
           scheduler = Scheduler(...)
           plans = scheduler.generate_task_and_schedule(...)
           
           # ... 仿真执行 ...
       
       # 清理
       qos_client.clear_all()
       qos_client.close()

5. 错误处理建议：
   
   try:
       qos_client.apply_rules_from_config(config_path, verify=True, dry_run=False)
   except RouterQoSException as e:
       logger.warning(f"路由器控制失败（仿真继续）: {e}")
       # 继续进行纯仿真（Communicator.simulate_bw = True）
   except Exception as e:
       logger.error(f"意外错误: {e}")

"""


if __name__ == "__main__":
    # 配置
    ROUTER_IP = "192.168.10.1"
    ROUTER_USER = "root"
    ROUTER_PASSWORD = "wslhy110"  # ⚠️ 实际使用时应从环境变量或密钥文件读取
    
    print("\n" + "=" * 60)
    print("Router QoS 集成示例")
    print("=" * 60)
    
    # 选择要运行的示例
    import sys
    if len(sys.argv) > 1:
        example_id = sys.argv[1]
    else:
        example_id = "menu"
    
    if example_id == "1":
        example_single_rule(ROUTER_IP, ROUTER_USER, ROUTER_PASSWORD)
    elif example_id == "2":
        example_batch_from_config(ROUTER_IP, ROUTER_USER, ROUTER_PASSWORD, 
                                 "config/network_config.json")
    elif example_id == "3":
        example_dry_run(ROUTER_IP, ROUTER_USER, ROUTER_PASSWORD)
    elif example_id == "4":
        example_cleanup(ROUTER_IP, ROUTER_USER, ROUTER_PASSWORD)
    elif example_id == "5":
        integration_with_main_scheduler(ROUTER_IP, ROUTER_USER, ROUTER_PASSWORD)
    else:
        print("""
使用方法:
  python examples_router_qos.py <example_id>

示例:
  1  - 添加单条规则
  2  - 从配置文件批量应用
  3  - 干运行模式
  4  - 清除所有规则
  5  - 与 main.py/scheduler 集成演示
        """)
