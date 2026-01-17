import torch
import sys
import os
sys.path.append(os.getcwd())

from models.dag_wrappers import ResNet_DAG_Wrapper

def test_resnet_pipeline():
    print("=== 测试 ResNet18 分布式切分 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 初始化模型
    model = ResNet_DAG_Wrapper(device=device)
    
    print(f"\n模型总层数 (Stages): {len(model)}")
    # 打印层结构
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {type(layer)}")

    # 2. 准备数据
    # ResNet18 默认输入 224，但全卷积部分支持任意尺寸，直到最后的 FC
    img = torch.randn(1, 3, 224, 224).to(device)
    
    # 3. 模拟切分：在第 3 层 (Stage 2) 结束处切分
    split_point = 3
    
    # --- Node A (0 -> 3) ---
    print(f"\n[Node A] 计算 Layer 0-3 ...")
    pack_a = model.forward_slice(img, 0, split_point)
    print(f"Node A 输出形状: {pack_a['main'].shape}")
    print(f"Node A 缓存内容: {pack_a['cache']} (ResNet 应该是空的)")
    
    # --- 模拟传输 ---
    pack_b_in = pack_a 
    
    # --- Node B (3 -> End) ---
    print(f"\n[Node B] 计算 Layer 3-6 ...")
    pack_b = model.forward_slice(pack_b_in, split_point, len(model))
    
    final_res = pack_b['main']
    print(f"Node B 最终输出形状: {final_res.shape} (应该是 [1, 1000])")
    
    # 验证是否报错
    print("\n✅ ResNet 分布式仿真测试通过！")

if __name__ == "__main__":
    test_resnet_pipeline()