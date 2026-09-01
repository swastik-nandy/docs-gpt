from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from typing import Any, Iterator


_GPU_LOCK = threading.RLock()


def is_cuda_device(
    device: str | None,
) -> bool:
    if device is None:
        return False

    return device.strip().lower().startswith(
        "cuda"
    )


def resolve_device(
    requested_device: str | None,
) -> str:
    requested = (
        requested_device
        or "cpu"
    ).strip()

    if not is_cuda_device(requested):
        return requested or "cpu"

    try:
        import torch

    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    return requested


@contextmanager
def retrieval_gpu_lease(
    requested_device: str | None,
) -> Iterator[str]:
    active_device = resolve_device(
        requested_device
    )

    if not is_cuda_device(active_device):
        yield active_device
        return

    with _GPU_LOCK:
        yield active_device


def move_model_to_device(
    model_owner: Any,
    device: str,
) -> None:
    inner_model = getattr(
        model_owner,
        "model",
        None,
    )

    inner_to = getattr(
        inner_model,
        "to",
        None,
    )

    if callable(inner_to):
        inner_to(device)
        return

    owner_to = getattr(
        model_owner,
        "to",
        None,
    )

    if callable(owner_to):
        owner_to(device)
        return

    raise RuntimeError(
        "Model does not expose a supported "
        "device-transfer method."
    )


def offload_model_to_cpu(
    model_owner: Any,
    cuda_device: str,
) -> None:
    if not is_cuda_device(cuda_device):
        return

    try:
        import torch

        torch.cuda.synchronize(
            torch.device(cuda_device)
        )

    except Exception:
        pass

    move_model_to_device(
        model_owner,
        "cpu",
    )

    gc.collect()

    try:
        import torch

        torch.cuda.empty_cache()

    except Exception:
        pass
