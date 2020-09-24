from laia.callbacks import GPUStats


def test_parse_gpu_stats():
    gpu_stats = {
        0: {
            "memory.used": (0.0, "MB"),
            "memory.free": (0.0, "MB"),
            "utilization.memory": (0.0, "%"),
            "utilization.gpu": (0.0, "%"),
        },
        1: {
            "memory.used": (3287.0, "MB"),
            "memory.free": (4695.0, "MB"),
            "utilization.memory": (8.0, "@"),
            "utilization.gpu": (16.0, "%"),
        },
    }
    expected = {
        "GPU-0": "0/0MB, memory_time=0%, GPU_time=0%",
        "GPU-1": "3287/7982MB, memory_time=8@, GPU_time=16%",
    }
    assert GPUStats.parse_gpu_stats(gpu_stats) == expected
