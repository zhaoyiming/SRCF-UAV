
import torch
import torch.nn as nn
from textwrap import dedent

# 1. 导入 fvcore 用于 FLOPs 分析
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
except ImportError:
    print("fvcore is not installed. Please run: pip install fvcore")
    exit()

# 假设您的类在 rar_module.py 中
from radar_sampling import RangeAdaptiveRadarAggregation


def analyze_model(model: nn.Module, model_name: str, inputs: tuple):
    """
    Analyzes and prints both parameter and FLOPs statistics for a given model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        model_name (str): A descriptive name for the model configuration.
        inputs (tuple): A tuple of dummy input tensors for the model's forward pass.
    """
    
    # --- 基本信息 ---
    header = f"📊 Analysis for: {model_name} "
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # --- 1. 参数分析 ---
    param_details = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        param_details.append((name, num_params))
    
    max_name_len = max(len(name) for name, _ in param_details) if param_details else 0
    print("\n--- [1] Parameter Breakdown ---")
    print(f"{'Layer Name':<{max_name_len}} | {'Parameters':>15}")
    print("-" * (max_name_len + 18))
    for name, num_params in param_details:
        print(f"{name:<{max_name_len}} | {num_params:>15,}")
    print("-" * (max_name_len + 18))

    # --- 2. FLOPs 分析 ---
    print("\n--- [2] FLOPs Breakdown ---")
    total_flops = 0
    try:
        # 使用 fvcore 进行分析
        flops_analyzer = FlopCountAnalysis(model, inputs)
        total_flops = flops_analyzer.total()
        
        # 打印 fvcore 生成的详细表格
        print(flop_count_table(flops_analyzer, max_depth=4))

    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
        print("This may be due to operations not supported by fvcore.")

    # --- 3. 最终总结 ---
    summary = f"""
    --- [3] Final Summary ---
      - Total Parameters:     {total_params/1e6:.3f} M ({total_params:,})
      - Trainable Parameters: {trainable_params/1e6:.3f} M ({trainable_params:,})
      - Total GFLOPs:         {total_flops/1e9:.3f} G
    """
    print(dedent(summary))
    print("\n")


if __name__ == "__main__":
    # --- 定义模型超参数和模拟输入 ---
    # FLOPs 计算需要具体的输入尺寸，我们在此定义
    BATCH_SIZE = 16
    NUM_IMG_QUERIES = 900  # 图像查询点的数量
    NUM_RADAR_POINTS = 64 # 雷达点的数量
    EMBED_DIM = 256
    RADAR_DIM = 256
    NUM_HEADS = 8
    
    print(f"Analyzing models with Input Shapes: Batch={BATCH_SIZE}, ImgQueries={NUM_IMG_QUERIES}, RadarPoints={NUM_RADAR_POINTS}")
    
    # 创建模拟输入张量
    img_feat = torch.randn(BATCH_SIZE, NUM_IMG_QUERIES, EMBED_DIM)
    radar_feat = torch.randn(BATCH_SIZE, NUM_RADAR_POINTS, RADAR_DIM)
    img_xyz = torch.rand(BATCH_SIZE, NUM_IMG_QUERIES, 3) * 80.0
    radar_xyz = torch.rand(BATCH_SIZE, NUM_RADAR_POINTS, 3) * 80.0
    
    # 将输入打包成元组，以传递给 FlopCountAnalysis
    dummy_inputs = (img_feat, radar_feat, img_xyz, radar_xyz)

    # --- Case 1: 'dot' attention, 'add' fusion ---
    model_dot_add = RangeAdaptiveRadarAggregation(
        embed_dim=EMBED_DIM, radar_dim=RADAR_DIM, num_heads=NUM_HEADS, attn_type="dot", fuse="add"
    )
    analyze_model(model_dot_add, "RAR (attn_type='dot', fuse='add')", dummy_inputs)

    # --- Case 2: 'dot' attention, 'cat' fusion ---
    model_dot_cat = RangeAdaptiveRadarAggregation(
        embed_dim=EMBED_DIM, radar_dim=RADAR_DIM, num_heads=NUM_HEADS, attn_type="dot", fuse="cat"
    )
    analyze_model(model_dot_cat, "RAR (attn_type='dot', fuse='cat')", dummy_inputs)
    
    # --- Case 3: 'deform' attention, 'add' fusion ---
    model_deform_add = RangeAdaptiveRadarAggregation(
        embed_dim=EMBED_DIM, radar_dim=RADAR_DIM, num_heads=NUM_HEADS, attn_type="deform", fuse="add"
    )
    analyze_model(model_deform_add, "RAR (attn_type='deform', fuse='add')", dummy_inputs)
    
    # --- Case 4: 'deform' attention, 'cat' fusion ---
    model_deform_cat = RangeAdaptiveRadarAggregation(
        embed_dim=EMBED_DIM, radar_dim=RADAR_DIM, num_heads=NUM_HEADS, attn_type="deform", fuse="cat"
    )
    analyze_model(model_deform_cat, "RAR (attn_type='deform', fuse='cat')", dummy_inputs)


# analyze_memory_usage.py
# Description: Analyzes the GPU memory usage of the RangeAdaptiveRadarAggregation module.
# To Run: Ensure this file is in the same directory as rar_module.py
#         and execute `python analyze_memory_usage.py`
# -----------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# from textwrap import dedent

# # 假设您的类在 rar_module.py 中
# from radar_sampling import RangeAdaptiveRadarAggregation

# def format_mem_size(num_bytes: int) -> str:
#     """Formats bytes into a human-readable string (KB, MB, GB)."""
#     if abs(num_bytes) < 1024:
#         return f"{num_bytes} B"
#     elif abs(num_bytes) < 1024**2:
#         return f"{num_bytes/1024:.2f} KB"
#     elif abs(num_bytes) < 1024**3:
#         return f"{num_bytes/1024**2:.2f} MB"
#     else:
#         return f"{num_bytes/1024**3:.2f} GB"

# def analyze_memory(model: nn.Module, model_name: str, inputs: tuple):
#     """
#     Analyzes and prints the GPU memory usage for a model during inference and training.
#     """
#     header = f"💾 Memory Analysis for: {model_name} "
#     print("\n" + "=" * len(header))
#     print(header)
#     print("=" * len(header))

#     # --- 将模型和数据移动到GPU ---
#     device = torch.device("cuda")
#     model.to(device)
#     gpu_inputs = tuple(t.to(device) for t in inputs)
    
#     # --- 1. 静态显存占用 (模型参数 + 输入数据) ---
#     static_mem = torch.cuda.memory_allocated(device)
#     print(f"[1] Static Memory Usage (Model + Inputs): {format_mem_size(static_mem)}")

#     # --- 2. 前向传播 (Inference) 峰值显存 ---
#     # 清理缓存并重置峰值统计
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats(device)
    
#     # 模拟推理过程
#     with torch.no_grad():
#         start_mem_forward = torch.cuda.memory_allocated(device)
#         model(*gpu_inputs)
#         peak_mem_forward = torch.cuda.max_memory_allocated(device)

#     # 峰值增量即为中间激活值占用的显存
#     activation_mem = peak_mem_forward - start_mem_forward
#     print(f"[2] Peak Memory during Forward Pass (Activations): {format_mem_size(activation_mem)}")
#     print(f"    (Total peak was {format_mem_size(peak_mem_forward)})")

#     # --- 3. 训练 (Forward + Backward) 峰值显存 ---
#     # 需要优化器和损失函数来模拟训练
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     criterion = nn.L1Loss() # 任意一个损失函数即可

#     # 清理缓存和统计数据
#     del gpu_inputs # 释放之前的输入
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats(device)
    
#     # 重新加载数据
#     model.train()
#     gpu_inputs = tuple(t.to(device) for t in inputs)
    
#     # -- 完整训练步骤 --
#     optimizer.zero_grad()
    
#     # 前向传播
#     output = model(*gpu_inputs)
    
#     # 损失计算
#     # 创建一个与输出形状相同的虚拟目标
#     dummy_target = torch.randn_like(output)
#     loss = criterion(output, dummy_target)
    
#     # 反向传播
#     loss.backward()
    
#     # 优化器步骤（通常不增加峰值，但包含在完整流程中）
#     optimizer.step()
    
#     peak_mem_training = torch.cuda.max_memory_allocated(device)
#     print(f"[3] Peak Memory during a Full Training Step (Activations + Gradients): {format_mem_size(peak_mem_training)}")

#     # 清理，以便下一次分析
#     del model, optimizer, gpu_inputs, output, loss
#     torch.cuda.empty_cache()


# if __name__ == "__main__":
#     if not torch.cuda.is_available():
#         print("CUDA is not available. This script requires a GPU.")
#         exit()

#     # --- 定义模型超参数和模拟输入 ---
#     BATCH_SIZE = 16 # 增加批量以更好地观察显存变化
#     NUM_IMG_QUERIES = 900
#     NUM_RADAR_POINTS = 64
#     EMBED_DIM = 256
    
#     print(f"Analyzing models with Input Shapes: Batch={BATCH_SIZE}, ImgQueries={NUM_IMG_QUERIES}, RadarPoints={NUM_RADAR_POINTS}")

#     # 创建模拟CPU输入张量
#     img_feat = torch.randn(BATCH_SIZE, NUM_IMG_QUERIES, EMBED_DIM)
#     radar_feat = torch.randn(BATCH_SIZE, NUM_RADAR_POINTS, EMBED_DIM)
#     img_xyz = torch.rand(BATCH_SIZE, NUM_IMG_QUERIES, 3) * 80.0
#     radar_xyz = torch.rand(BATCH_SIZE, NUM_RADAR_POINTS, 3) * 80.0
#     cpu_inputs = (img_feat, radar_feat, img_xyz, radar_xyz)

#     # --- 逐个分析不同配置 ---
#     # Case 1: 'dot' attention
#     model_dot = RangeAdaptiveRadarAggregation(embed_dim=EMBED_DIM, attn_type="dot")
#     analyze_memory(model_dot, "RAR (attn_type='dot')", cpu_inputs)

#     # Case 2: 'deform' attention
#     model_deform = RangeAdaptiveRadarAggregation(embed_dim=EMBED_DIM, attn_type="deform")
#     analyze_memory(model_deform, "RAR (attn_type='deform')", cpu_inputs)