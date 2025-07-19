from .constants import *
from .utils import *
from .modules import *
from .messages import *
from dataclasses import dataclass, field
import edge_tts
import tempfile
import torchaudio
import jieba
import os
from time import time

voices = get_voices()
@dataclass
class TTSEdgeConfig(ModuleConfig):
    """使用微软的 TTS"""
    voice: str = field(default=voices[0]['voice'], metadata={
        "required": False,
        "desc": "选择声音",
        "selection": True,
        "options": [
            {"key": voice['friendly_name'], "value": voice['voice']}
            for voice in voices
        ]
    })

class TTSEdge(TTSBase):
    role: ModuleRoles = ModuleRoles.TTS
    config_class = TTSEdgeConfig
    config: config_class
    def __init__(self, config: config_class | None = None, **kwargs): # 为了同时支持传入 config 和传入单独配置项两种方式
        super().__init__(config, **kwargs)
        self.voice = self.config.voice
    
    async def generate_sentence(self, id: str, content: str, emotions: dict[str, float]) -> TTSAlignedAudio:
        communicate = edge_tts.Communicate(content, self.voice)
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_name = os.path.join(temp_dir, f"voice{time()}.wav")
            await communicate.save(audio_name)
            info = torchaudio.info(audio_name)
            duration = info.num_frames / info.sample_rate
            words = [*jieba.cut(content)]
            intervals = [
                {"token": word, "duration": duration / len(words)}
                for word in words
            ]
            with open(audio_name, 'rb') as f:
                audio_data = f.read()
        return TTSAlignedAudio(self, id, audio_data, intervals)
