import torch
import torch.utils.benchmark as bench

def matmul_time(size: int, device: str, num_runs: int = 50):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    timer = bench.Timer(
        stmt="c = a @ b; torch.cuda.synchronize() if a.is_cuda else None",
        globals={"a": a, "b": b, "torch": torch}
    )
    return timer.blocked_autorange(min_run_time=0.5).median

sizes = [64, 128, 256, 512, 1024, 2048, 4096]
cpu_times, gpu_times = [], []

for n in sizes:
    cpu_times.append(matmul_time(n, "cpu"))
    gpu_times.append(matmul_time(n, "cuda"))

for n, cpu_t, gpu_t in zip(sizes, cpu_times, gpu_times):
    print(f"{n:>5}×{n:>5} | CPU {cpu_t*1e3:7.1f} ms | GPU {gpu_t*1e3:7.1f} ms "
          f"| speed-up ×{cpu_t/gpu_t:4.1f}")
