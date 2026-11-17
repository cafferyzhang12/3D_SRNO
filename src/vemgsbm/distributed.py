"""Small torchrun helpers shared by distributed training and case inference."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    owns_group: bool = False

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def is_distributed(self):
        return self.world_size > 1

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def broadcast_object(self, value, src=0):
        if not self.is_distributed:
            return value
        values = [value if self.rank == src else None]
        dist.broadcast_object_list(values, src=src)
        return values[0]

    def check_errors(self, local_error):
        """Raise on every rank if any rank reports an error before a collective."""
        if not self.is_distributed:
            if local_error:
                raise RuntimeError(str(local_error))
            return
        failed = torch.tensor(int(bool(local_error)), dtype=torch.int32, device=self.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            errors = [None] * self.world_size
            dist.all_gather_object(errors, str(local_error) if local_error else None)
            details = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
            raise RuntimeError("Distributed operation failed. " + " | ".join(details))

    def main_call(self, function, broadcast_result=False):
        """Execute on rank zero and send success or failure to all participants."""
        if not self.is_distributed:
            return function()
        result, status = None, None
        if self.is_main:
            try:
                result = function()
            except Exception as error:
                status = f"{type(error).__name__}: {error}"
        status = self.broadcast_object(status)
        if status is not None:
            raise RuntimeError("Rank-zero operation failed. " + status)
        if broadcast_result:
            result = self.broadcast_object(result)
        return result

    def sum(self, values):
        totals = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.is_distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return totals.cpu().tolist()

    def assert_same(self, value, description):
        if self.is_distributed:
            values = [None] * self.world_size
            dist.all_gather_object(values, value)
            if any(item != values[0] for item in values[1:]):
                raise ValueError(f"Distributed ranks disagree on {description}")

    def cleanup(self):
        # Failure cleanup deliberately has no barrier, so a failed peer cannot
        # prevent the launcher from tearing down the remaining workers.
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_group = False


def initialize_distributed(device_name="auto", timeout_minutes=60):
    """Honor torchrun rank variables and bind each CUDA worker to LOCAL_RANK."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    existing = dist.is_available() and dist.is_initialized()
    if existing:
        world_size, rank = dist.get_world_size(), dist.get_rank()
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun RANK/WORLD_SIZE/LOCAL_RANK configuration")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if world_size > 1:
            device = torch.device("cuda", local_rank)
        elif device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Distributed execution supports CPU or CUDA devices")
    owns_group = False
    if world_size > 1 and not existing:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=world_size,
            timeout=timedelta(minutes=timeout_minutes),
        )
        owns_group = True
    return DistributedContext(device, rank, world_size, local_rank, owns_group)


class ValidationShardSampler(Sampler):
    """Shard fixed validation indices without repeating or dropping any item."""

    def __init__(self, dataset, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid validation shard")
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
