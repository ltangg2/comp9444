"""Run one of Lily's PullNet scripts unchanged, but hand unused GPU memory back after every chunk.

Why this exists: on this PC (8 GB card, ~1.4 GB of it used by Windows) PyTorch kept more and more
reserved-but-unused GPU memory after each training chunk (live data stayed ~94 MB, reserved grew
past 5.7 GB). Once the card was full, Windows spilled into normal RAM and each chunk went from
~15 s to ~19 min, so the first attempt hit its 5-hour limit without saving a model
(logs/train_hop3_FAILED_gpu_memory.log).

train.py already calls gc.collect() once per chunk; this adds torch.cuda.empty_cache() to that
call. It does not change the model, the data or the training maths.

Usage (from analysis/pullnet_run/):  python run_with_gpu_cleanup.py ../../pullnet/train.py --hop 3
"""
import gc
import os
import runpy
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation
import torch  # noqa: E402

_collect = gc.collect


def collect_and_free_gpu(*args, **kwargs):
    result = _collect(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


gc.collect = collect_and_free_gpu
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
