import os
import sys
from typing import List, Tuple

# Ensure synchronous CUDA error reporting
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch

from reference import generate_input, check_implementation
from submission import custom_kernel


TESTS = [
    {"m": [96, 128], "n": [128, 256], "k": [256, 512], "g": 2, "seed": 1111},
    {"m": [256, 72], "n": [512, 384], "k": [256, 256], "g": 2, "seed": 1111},
    {"m": [128, 128], "n": [128, 256], "k": [512, 256], "g": 2, "seed": 1111},
    {"m": [80, 128, 256], "n": [384, 256, 128], "k": [256, 512, 256], "g": 3, "seed": 1111},
    {"m": [64, 72, 96], "n": [128, 384, 512], "k": [512, 512, 256], "g": 3, "seed": 1111},
    {"m": [64, 256, 128], "n": [768, 128, 256], "k": [512, 256, 512], "g": 3, "seed": 1111},
    {"m": [128, 128, 64], "n": [256, 512, 512], "k": [768, 256, 768], "g": 3, "seed": 1111},
    {"m": [128, 128, 128, 128], "n": [128, 128, 128, 128], "k": [512, 256, 512, 256], "g": 4, "seed": 1111},
    {"m": [40, 56, 384, 512], "n": [512, 384, 256, 128], "k": [256, 256, 256, 256], "g": 4, "seed": 1111},
    {"m": [512, 384, 256, 128], "n": [256, 256, 256, 256], "k": [512, 768, 512, 768], "g": 4, "seed": 1111},
]

BENCHMARKS = [
    {"m": [80, 176, 128, 72, 64, 248, 96, 160], "n": [4096] * 8, "k": [7168] * 8, "g": 8, "seed": 1111},
    {"m": [40, 76, 168, 72, 164, 148, 196, 160], "n": [7168] * 8, "k": [2048] * 8, "g": 8, "seed": 1111},
    {"m": [192, 320], "n": [3072, 3072], "k": [4096, 4096], "g": 2, "seed": 1111},
    {"m": [128, 384], "n": [4096, 4096], "k": [1536, 1536], "g": 2, "seed": 1111},
]


def _run_case(case: dict, label: str, check: bool) -> None:
    data = generate_input(tuple(case["m"]), tuple(case["n"]), tuple(case["k"]), case["g"], case["seed"])
    if check:
        ok, msg = check_implementation(custom_kernel, data)
        if not ok:
            raise RuntimeError(f"{label}: correctness check failed: {msg}")
    else:
        custom_kernel(data)
    torch.cuda.synchronize()
    print(f"{label}: OK")


def main(argv: List[str]) -> int:
    mode = "all"
    check = True
    if "--benchmarks" in argv:
        mode = "benchmarks"
    if "--tests" in argv:
        mode = "tests"
    if "--no-check" in argv:
        check = False

    if mode in ("all", "tests"):
        for idx, case in enumerate(TESTS, start=1):
            _run_case(case, f"test[{idx}]", check)
    if mode in ("all", "benchmarks"):
        for idx, case in enumerate(BENCHMARKS, start=1):
            _run_case(case, f"benchmark[{idx}]", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
