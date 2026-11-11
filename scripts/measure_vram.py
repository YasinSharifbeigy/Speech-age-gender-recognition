"""Measure PyTorch CUDA allocator peaks for one real recording."""
import argparse
import json
from pathlib import Path

import torch

from oragen_service.engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use the CUDA image with --gpus all.")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    result = Engine("cuda").predict(args.audio)
    torch.cuda.synchronize()
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "note": "PyTorch allocator only; driver/context and other processes add overhead.",
        "result": result,
    }, indent=2))


if __name__ == "__main__":
    main()
