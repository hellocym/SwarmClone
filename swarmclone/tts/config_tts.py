from dataclasses import dataclass

@dataclass
class TTSConfig:
    MODEL_PATH: str = "coqui/XTTS-v2"
    REFERENCE_WAV_PATH: str = "coqui/XTTS-v2/samples/zh-cn-sample.wav"
