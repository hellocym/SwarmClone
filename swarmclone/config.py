from dataclasses import dataclass, field

@dataclass
class Config:
    VOCAB_SIZE: int = 32768
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2})
