from enum import Enum
from dataclasses import dataclass


class Role(Enum):
    USER: str = "user"
    SYSTEM: str = "system"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return dict(role=self.role.value, content=self.content)
