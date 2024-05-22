from enum import Enum
from dataclasses import dataclass, asdict


class Role(Enum):
    USER: str = "user"
    SYSTEM: str = "system"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def __str__(self) -> str:
        return asdict(self)

    def __repr__(self) -> str:
        return asdict(self)
