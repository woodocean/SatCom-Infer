# Router QoS API 快速参考 (极简方案A版本)

这个文档总结了 `RouterQoSClient` (位于 `core/router_qos.py`) 最小集实现版本的所有主要接口、使用场景、参数说明与集成指南。

---

## 核心类：RouterQoSClient

```python
class RouterQoSClient:
    """
    轻量级远程 OpenWrt 路由器 QoS 控制客户端
    
    初始化：
        qos = RouterQoSClient(
            host='192.168.10.1',      # 路由器 IP
            username='root',          # SSH 用户名
            password='password',      # SSH 密码
            port=22,                  # SSH 端口（默认 22）
            ssh_timeout=10            # SSH 超时秒数（默认 10）
        )
    
    上下文管理器（推荐）：
        with RouterQoSClient(...) as qos:
            qos.apply_rules_from_config(...)
            # 退出时自动自动调用 close() 关闭 SSH 连接
    """
```

---

## 主要 API 方法

### 1. `add_bandwidth_limit_with_delay`
添加带宽限制 + 延迟/抖动/丢包 规则。

**签名：**
```python
def add_bandwidth_limit_with_delay(self, src_ip, dst_ip, src_port, dst_port,
                                   rate_mbit, delay_ms=0, jitter_ms=0, loss_percent=0,
                                   protocol='tcp', dry_run=False):
```

**功能：**
- 自动根据 `dst_ip` 解析对应的 VLAN 接口 (如 `br-lan.2`, `br-lan.3` 等)。
- 自动基于五元组哈希生成唯一的 `mark` 和 `classid`，无需手动追踪维护优先队列。
- 调用 `iptables` 进行网络层流量打标，调用 `tc htb` 和 `tc filter` 限制应用带宽。
- 自动在 `htb` 队列下挂载 `netem` 队列子分支，顺滑注入网络传输的 `delay_ms`、`jitter_ms` 和 `loss_percent` 值。

**返回值：** 成功返回 `{'ok': True, 'mark': mark, 'classid': classid}`，失败返回 `{'ok': False, 'error': "错误信息"}`

---

### 2. `remove_rule`
根据给定的 mark 和五元组信息精确定向移除某个网络配置规则。

**签名：**
```python
def remove_rule(self, mark, src_ip, dst_ip, src_port, dst_port, protocol='tcp'):
```

**功能：**
- 删除与五元组精确匹配的 `iptables` mangle 转发规则。
- 删除对应的 `tc filter`, `tc qdisc` (netem层) 和 `tc class` (htb树形级别层)。

**返回值：** `{'ok': True}`

---

### 3. `apply_rules_from_config`
直接从 JSON 配置文件中读取网络链路拓扑逻辑，批量应用所有的 QoS 网络约束环境。

**签名：**
```python
def apply_rules_from_config(self, config_path, dry_run=False):
```

**关联示例数据 (`config/network_config.json`)：**
```json
{
  "links": {
    "Sat_1_to_GS": {
      "src_ip": "192.168.10.100",
      "dst_ip": "192.168.3.100",
      "src_port": 6000,
      "dst_port": 5000,
      "bandwidth_mbps": 100.0,
      "propagation_ms": 50.0,
      "jitter_ms": 5.0,
      "protocol": "tcp"
    }
  }
}
```

**功能：**
- 快速读取并解析类似于 `network_config.json` 的标准映射文件。
- 自动化遍历并加载所有物理或虚拟 `links`，抽取 `bandwidth_mbps` 和 `propagation_ms` 字段执行。
- 在部署结束后将运行状态更新覆写进配置文件中保存同步。

**返回值：** `{'ok': bool, 'applied': int, 'failed': int}`

---

### 4. `clear_all`
一键环境清除功能（测试重置或环境归零时使用）。

**签名：**
```python
def clear_all(self, dry_run=False):
```

**功能：**
- 通过 `iptables -t mangle -F` 清理全部五元组流表的流量标记。
- 在主要的路由网关虚拟交换出口 (如 `br-lan`, `br-lan.2`, `br-lan.4`) 端清除残留的 `qdisc root` 分支。

**返回值：** 成功返回 `{'ok': True}`，如果产生终端连接出错则返回 `{'ok': False}`。