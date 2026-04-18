"""
Router QoS Control - 最小实现版本（~200 行）
轻量级远程路由器带宽 + 延迟控制
"""

import paramiko
import json
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('RouterQoS')


class RouterQoSClient:
    """轻量级远程 OpenWrt 路由器 QoS 控制客户端"""
    
    MARK_RANGE = (1000, 1999)
    
    def __init__(self, host, username, password, port=22, ssh_timeout=10):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.ssh_timeout = ssh_timeout
        self.ssh = None
    
    def _exec(self, cmd):
        """执行 SSH 命令，返回 (returncode, stdout, stderr)"""
        if self.ssh is None:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(self.host, self.port, self.username, self.password, timeout=self.ssh_timeout)
        
        _, stdout, stderr = self.ssh.exec_command(cmd, timeout=self.ssh_timeout)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    
    def _get_iface(self, dst_ip):
        """根据目标 IP 推断接口"""
        if dst_ip.startswith('192.168.10.'):
            return 'br-lan.2'
        elif dst_ip.startswith('192.168.3.'):
            return 'br-lan.4'
        elif dst_ip.startswith('192.168.2.'):
            return 'br-lan.3'
        return 'br-lan'
    
    def _mark(self, src_ip, dst_ip, src_port, dst_port, proto='tcp'):
        """计算五元组的 Mark ID"""
        key = f"{src_ip}:{src_port}→{dst_ip}:{dst_port}/{proto}"
        h = hash(key) % (self.MARK_RANGE[1] - self.MARK_RANGE[0])
        return self.MARK_RANGE[0] + h
    
    def add_bandwidth_limit_with_delay(self, src_ip, dst_ip, src_port, dst_port,
                                       rate_mbit, delay_ms=0, jitter_ms=0, loss_percent=0,
                                       protocol='tcp', dry_run=False):
        """添加带宽限制 + 延迟规则"""
        mark = self._mark(src_ip, dst_ip, src_port, dst_port, protocol)
        iface = self._get_iface(dst_ip)
        classid = mark
        prio = 2
        
        logger.info(f"规则: {src_ip}:{src_port}→{dst_ip}:{dst_port} "
                   f"{rate_mbit}Mbps/{delay_ms}ms (mark={mark})")
        
        if dry_run:
            return {'ok': True, 'mark': mark, 'classid': classid}
        
        try:
            # 1. 确保根 qdisc 存在
            self._exec(f"tc qdisc show dev {iface} | grep -q htb || "
                      f"(tc qdisc add dev {iface} root handle 1: htb default 30 && "
                      f"tc class add dev {iface} parent 1: classid 1:1 htb rate 1000mbit)")
            
            # 2. iptables mark
            self._exec(f"iptables -t mangle -A FORWARD -s {src_ip} -d {dst_ip} "
                       f"-p {protocol} --sport {src_port} --dport {dst_port} "
                       f"-j MARK --set-mark {mark}")
            
            # 3. tc class
            self._exec(f"tc class add dev {iface} parent 1:1 classid 1:{classid} "
                       f"htb rate {rate_mbit}mbit 2>/dev/null")
            
            # 4. tc filter
            self._exec(f"tc filter add dev {iface} protocol ip parent 1:0 prio {prio} "
                       f"handle {mark} fw flowid 1:{classid} 2>/dev/null")
            
            # 5. netem（如需延迟）
            if delay_ms > 0 or loss_percent > 0:
                netem = f"delay {delay_ms}ms"
                if jitter_ms > 0:
                    netem += f" {jitter_ms}ms"
                if loss_percent > 0:
                    netem += f" loss {loss_percent}%"
                
                self._exec(f"tc qdisc add dev {iface} parent 1:{classid} handle {mark}: "
                          f"netem {netem}")
            
            logger.info(f"✓ 已应用")
            return {'ok': True, 'mark': mark, 'classid': classid}
        
        except Exception as e:
            logger.error(f"✗ 失败: {e}")
            return {'ok': False, 'error': str(e)}
    
    def remove_rule(self, mark, src_ip, dst_ip, src_port, dst_port, protocol='tcp'):
        """移除规则"""
        iface = self._get_iface(dst_ip)
        classid = mark
        
        logger.info(f"移除: mark={mark}")
        
        self._exec(f"iptables -t mangle -D FORWARD -s {src_ip} -d {dst_ip} "
                   f"-p {protocol} --sport {src_port} --dport {dst_port} "
                   f"-j MARK --set-mark {mark} 2>/dev/null")
        self._exec(f"tc filter del dev {iface} parent 1:0 handle {mark} fw 2>/dev/null")
        self._exec(f"tc qdisc del dev {iface} parent 1:{classid} 2>/dev/null")
        self._exec(f"tc class del dev {iface} classid 1:{classid} 2>/dev/null")
        
        logger.info(f"✓ 已移除")
        return {'ok': True}
    
    def apply_rules_from_config(self, config_path, dry_run=False):
        """从 JSON 配置批量应用规则"""
        logger.info(f"读取配置: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"读取失败: {e}")
            return {'ok': False, 'applied': 0, 'failed': 0}
        
        links = config.get('links', {})
        applied = failed = 0
        
        for link_name, link in links.items():
            try:
                result = self.add_bandwidth_limit_with_delay(
                    src_ip=link['src_ip'],
                    dst_ip=link['dst_ip'],
                    src_port=link.get('src_port', 6000),
                    dst_port=link.get('dst_port', 5000),
                    rate_mbit=link.get('bandwidth_mbps', 100),
                    delay_ms=link.get('propagation_ms', 0),
                    jitter_ms=link.get('jitter_ms', 0),
                    loss_percent=link.get('loss_percent', 0),
                    protocol=link.get('protocol', 'tcp'),
                    dry_run=dry_run
                )
                
                if result['ok']:
                    applied += 1
                    link['applied'] = True
                    link['mark'] = result['mark']
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"{link_name} 失败: {e}")
                failed += 1
        
        # 写回配置
        if not dry_run:
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            except:
                pass
        
        logger.info(f"完成: {applied} 成功, {failed} 失败")
        return {'ok': failed == 0, 'applied': applied, 'failed': failed}
    
    def clear_all(self, dry_run=False):
        """清除所有规则"""
        logger.warning("清除所有规则")
        
        if dry_run:
            return {'ok': True}
        
        try:
            self._exec("iptables -t mangle -F")
            for iface in ['br-lan', 'br-lan.1', 'br-lan.2', 'br-lan.3', 'br-lan.4']:
                self._exec(f"tc qdisc del dev {iface} root 2>/dev/null")
            
            logger.info("✓ 已清除")
            return {'ok': True}
        except Exception as e:
            logger.error(f"清除失败: {e}")
            return {'ok': False}
    
    def close(self):
        """关闭连接"""
        if self.ssh:
            try:
                self.ssh.close()
            except:
                pass
            self.ssh = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
