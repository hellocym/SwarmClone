from dataclasses import dataclass, field

@dataclass
class Config:
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2})
    NUM_WORKERS: int = 4
    DEVICE: str = "cuda"

    # 网络配置
    PANEL_HOST: str = "localhost"
    PANEL_WEB_PORT: int = 8080
    PANEL_TO_LLM: int = 8000
    PANEL_TO_TTS: int = 8001
    PANEL_TO_UNITY: int = 8002
    PANEL_FROM_LLM: int = 8003
    PANEL_FROM_ASR: int = 8004
    PANEL_FROM_CHAT: int = 8005
