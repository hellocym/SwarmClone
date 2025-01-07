from dataclasses import dataclass

@dataclass
class TTSConfig:
    MODEL_PATH: str = "~/.swarmclone/tts/coqui/XTTS-v2"
    REFERENCE_WAV_PATH: str = f"{MODEL_PATH}/samples/zh-cn-sample.wav"
