from dataclasses import dataclass, field

@dataclass
class Config:
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2})
    NUM_WORKERS: int = 4
    DEVICE: str = "cuda"

    # 网络配置
    LLM_HOST: str = "localhost"
    LLM_TO_PANEL: int = 1000
    LLM_FROM_PANEL: int = 1001

    ASR_HOST: str = "localhost"
    ASR_TO_PANEL: int = 2000

    TTS_HOST: str = "localhost"
    TTS_FROM_PANEL: int = 3000

    CHAT_HOST: str = "localhost"
    CHAT_TO_PANEL: int = 4000

    UNITY_HOST: str = "localhost"
    UNITY_FROM_PANEL: int = 5000

    PANEL_HOST: str = "localhost"
    PANEL_TO_LLM: int = 6000
    PANEL_TO_TTS: int = 6001
    PANEL_TO_UNITY: int = 6002
    PANEL_FROM_LLM: int = 6003
    PANEL_FROM_ASR: int = 6004
    PANEL_FROM_CHAT: int = 6005
