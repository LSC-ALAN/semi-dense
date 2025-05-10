import torch
import time

# 选择 GPU 1（即索引为 1 的 GPU）
device = torch.device("cuda:1")

# 设置目标显存为 15GB
max_memory = 15 * 1024 ** 3  # 15GB

# 获取 GPU 1 的总显存
total_memory = torch.cuda.get_device_properties(device).total_memory

# 设置 PyTorch 允许占用的显存比例
memory_fraction = max_memory / total_memory

# 无限循环，保持显存占用
while True:
    try:
        # 设置最大显存占用比例
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)

        # 打印当前显存使用情况
        allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 3  # GB
        cached_memory = torch.cuda.memory_reserved(device) / 1024 ** 3  # GB
        print(f"Allocated memory: {allocated_memory:.2f} GB")
        print(f"Reserved memory: {cached_memory:.2f} GB")

        # 模拟显存占用，创建多个大张量来消耗显存
        for _ in range(100):  # 循环创建多个较大的张量
            x = torch.randn(1000, 1024, 1024, device=device)  # 创建更大的张量
            del x  # 删除张量，避免内存泄漏

        time.sleep(1)  # 暂停 1 秒钟继续执行

    except KeyboardInterrupt:
        print("Program interrupted.")
        break
    except Exception as e:
        print(f"Error: {e}")
        break
