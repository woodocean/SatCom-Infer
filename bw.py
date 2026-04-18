import paramiko
import time
import platform
import subprocess

class RouterQoS:
    def __init__(self, host, username, password, port=22):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.ssh = None

    def _connect(self):
        if not self.ssh:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(self.host, self.port, self.username, self.password)

    def _exec(self, command):
        self._connect()
        stdin, stdout, stderr = self.ssh.exec_command(command)
        return stdout.read().decode(errors="replace")

    def _get_out_interface(self, dst_ip):
        if dst_ip.startswith('192.168.10.'): return 'br-lan.2'
        elif dst_ip.startswith('192.168.3.'): return 'br-lan.4'
        elif dst_ip.startswith('192.168.2.'): return 'br-lan.3'
        return 'br-lan'

    def clear_all(self):
        print("[INFO] 正在清除所有防火墙标记和流量控制规则...")
        self._exec("iptables -t mangle -F")
        for iface in ['br-lan', 'br-lan.1', 'br-lan.2', 'br-lan.3', 'br-lan.4']:
            self._exec(f"tc qdisc del dev {iface} root 2>/dev/null")
        print("[OK] 环境已归零。")

    def add_delay(self, dst_ip, delay_ms, limit=100000):
        """
        全速模式：不设 htb class 限制，直接在接口根部挂载 netem 延迟。
        这会让该接口所有去往该方向的流量统一增加延迟，且理论上能跑满物理带宽。
        """
        iface = self._get_out_interface(dst_ip)
        print(f"[INFO] 正在为接口 {iface} (目标: {dst_ip}) 设置 {delay_ms}ms 延迟...")
        
        # 1. 确保环境干净
        self._exec(f"tc qdisc del dev {iface} root 2>/dev/null")
        
        # 2. 直接添加 netem 作为根队列 (不使用 htb)
        # 提高 limit 到 100,000 确保在大带宽+高延迟下不丢包
        cmd = f"tc qdisc add dev {iface} root netem delay {delay_ms}ms limit {limit}"
        self._exec(cmd)
        
        print(f"[OK] 延迟已生效。请执行 iperf3 测试。")

    def close(self):
        if self.ssh: self.ssh.close()

if __name__ == "__main__":
    # 配置
    ROUTER_IP = "192.168.10.1"
    USER = "root"
    PASSWORD = "wslhy110"
    TARGET_IP = "192.168.3.181"

    qos = RouterQoS(ROUTER_IP, USER, PASSWORD)
    
    # 1. 先清空
    qos.clear_all()
    
    # 2. 统一加 100ms 延迟 (不限速)
    qos.add_delay(TARGET_IP, delay_ms=100)
    
    qos.close()