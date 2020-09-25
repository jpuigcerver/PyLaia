from laia.callbacks import ProgressBarGPUStats


def test_parse_gpu_stats():
    gpu_ids = "0,1"
    gpu_stats = [[0.0, 0.0, 0.0, 0.0], [3287.3, 4695.0, 8.0, 16]]
    gpu_stat_keys = [
        ("memory.used", "MB"),
        ("memory.free", "MB"),
        ("utilization.memory", "%"),
        ("utilization.gpu", "%"),
    ]
    expected = {"GPU-0": "0/0MB", "GPU-1": "3287/7982MB"}
    assert (
        ProgressBarGPUStats.parse_gpu_stats(gpu_ids, gpu_stats, gpu_stat_keys)
        == expected
    )
