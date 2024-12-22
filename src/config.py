from dataclasses import dataclass

# 设置
@dataclass
class Config:
    SPECIAL_TOKENS: list[str] = ["<eos>", "<pad>", "<unk>"]
