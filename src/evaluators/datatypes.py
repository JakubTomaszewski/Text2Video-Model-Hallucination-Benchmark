import torch
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def __call__(self, prompt: str, frames: torch.Tensor, **kwargs) -> float:
        pass
