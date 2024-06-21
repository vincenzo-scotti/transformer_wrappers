from typing import Dict, Type

from torch.optim import Optimizer as Optimiser
import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from bitsandbytes.optim import PagedAdamW


optimizer_mapping: Dict[str, Type[Optimiser]] = {
    torch.optim.AdamW.__name__: torch.optim.AdamW,
    torch.optim.RMSprop.__name__: torch.optim.RMSprop,
    PagedAdamW.__name__: PagedAdamW
}

lr_scheduler_mapping: Dict[str, Type[LRScheduler]] = {
    torch.optim.lr_scheduler.OneCycleLR.__name__: torch.optim.lr_scheduler.OneCycleLR
}
