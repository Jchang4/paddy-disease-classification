import traceback

import torch
from fastai.callback.all import Callback

from .gpu_memory import clean_memory

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class ToDeviceCallback(Callback):
    def before_epoch(self):
        self.learn.model = self.learn.model.to(device)

    def before_batch(self):
        self.learn.xb = tuple(xb.to(device) for xb in self.learn.xb)
        self.learn.yb = tuple(yb.to(device) for yb in self.learn.yb)


class safely_train_with_gpu:
    "context manager to reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            return True
        traceback.clear_frames(exc_tb)
        clean_memory()
        raise exc_type(exc_val).with_traceback(exc_tb) from None
