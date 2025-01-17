from dataclasses import dataclass

@dataclass
class TTSConfig:
    MODEL_PATH: str = "~/.swarmclone/tts/coqui/XTTS-v2"
    REFERENCE_WAV_PATH: str = "samples/zh-cn-sample.wav" # 相对于MODEL_PATH的路径
