import subprocess
import time
import re

def start_clumsy(filter_rule, throttle_kbps, duration_sec=None):
    clumsy_path = r"D:\Microsoft_download\clumsy-0.3-win64-a\clumsy.exe"
    cmd = [
        clumsy_path,
        "--filter", filter_rule,
        "--bandwidth", "on",
        "--bandwidth-bandwidth", str(throttle_kbps),
        "--bandwidth-inbound", "on",
        "--bandwidth-outbound", "on"
    ]
    # 隐藏窗口，以管理员权限运行
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW)
    if duration_sec:
        time.sleep(duration_sec)
        proc.terminate()
        return None
    return proc

def run_iperf3_test(target_ip, duration=20, reverse=False):
    """运行 iperf3 测试并返回平均带宽（Mbps），失败返回 None"""
    iperf_path = r"D:\Microsoft_download\iperf3.1.4_64\iperf3.exe"  # 请修改为你的 iperf3 路径
    cmd = [iperf_path, "-c", target_ip, "-t", str(duration)]
    if reverse:
        cmd.append("-R")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration+10)
    except subprocess.TimeoutExpired:
        print("iperf3 超时")
        return None

    # 解析输出，寻找 sender 的带宽行
    output = result.stdout + result.stderr
    # 匹配类似 "0.00-5.00 sec  132 MBytes  221 Mbits/sec  sender" 的行
    # 或者 "0.00-5.00 sec  132 MBytes  221 Mbits/sec"
    # 捕获带宽数值
    pattern = r'(\d+\.?\d*)\s*Mbits/sec'
    matches = re.findall(pattern, output)
    if matches:
        # 取最后一个匹配（通常是总带宽）
        bw = float(matches[-1])
        return bw
    else:
        print("无法解析 iperf3 输出：")
        print(output[-500:])  # 打印最后500字符
        return None

if __name__ == "__main__":
    JETSON_IP = "192.168.0.106"
    # 1. 启动限速（双向，10 Mbps = 1250 KB/s）
    filter_rule = f"ip.DstAddr == {JETSON_IP} or ip.SrcAddr == {JETSON_IP}"
    print("[*] 启动 Clumsy 限速 80 Mbps...")
    proc = start_clumsy(filter_rule, throttle_kbps=10240, duration_sec=None)

    # 等待规则生效
    time.sleep(2)

    # 2. 运行正向测试（PC -> Jetson）
    print("[*] 测试正向带宽 (PC -> Jetson):")
    bw = run_iperf3_test(JETSON_IP, duration=5, reverse=False)
    if bw is not None:
        print(f"    -> 带宽: {bw:.2f} Mbps")
    else:
        print("    -> 测试失败")

    # 3. 运行反向测试（Jetson -> PC）
    print("[*] 测试反向带宽 (Jetson -> PC):")
    bw = run_iperf3_test(JETSON_IP, duration=5, reverse=True)
    if bw is not None:
        print(f"    -> 带宽: {bw:.2f} Mbps")
    else:
        print("    -> 测试失败")

    # 4. 停止 Clumsy
    print("[*] 停止限速...")
    if proc:
        proc.terminate()
        proc.wait()