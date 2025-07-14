import asyncio
import numpy as np
import json
from typing import Any
from dataclasses import dataclass, field

from .sherpa_asr import create_recognizer
from ..modules import *
from ..messages import ASRMessage, ASRActivated

@dataclass
class ASRSherpaConfig(ModuleConfig):
    host: str = field(default="0.0.0.0")
    port: int = field(default=8004)
    userdb: dict[str, str] = field(default_factory=lambda : {"DeveloperA": "12345"})
    model: str = field(default="zipformer")
    quantized: str = field(default="fp32")
    model_path: str = field(default="~/.swarmclone/asr/")
    decoding_method: str = field(default="greedy_search")
    provider: str = field(default="cpu")
    hotwords_file: str = field(default="")
    hotwords_score: float = field(default=1.5)
    blank_penalty: float = field(default=0.0)
    vadmodel_path: str = field(default="~/.swarmclone/vad/")

class ASRSherpa(ModuleBase):
    role: ModuleRoles = ModuleRoles.ASR
    config_class = ASRSherpaConfig
    def __init__(self, config: ASRSherpaConfig | None = None, **kwargs):
        super().__init__()
        self.config = self.config_class(**kwargs) if config is None else config
        self.recognizer = create_recognizer(self.config)
        self.stream = self.recognizer.create_stream()
        self.sample_rate = 16000
        self.samples_per_read = int(0.1 * self.sample_rate)
        self.userdb = self.config.userdb
        self.clientdict = {}
        self.server = None

    async def run(self):
        self.server = await asyncio.start_server(
            self.handle_client,
            self.config.host,
            self.config.port
        )
        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
        try:
            check = await reader.read(1024)
            checkmessage = json.loads(check.decode(), object_hook=dict[str, Any])
        except(UnicodeDecodeError):
            print("不是可接受的鉴权信息")
            return None
        if (password := self.userdb.get(checkmessage['name'])) is None:
            writer.write('WRUSR\n'.encode('utf-8'))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return None
        if not checkmessage['passwd'] == password:
            writer.write('WRPWD\n'.encode('utf-8'))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return None
        else:
            writer.write('OK\n'.encode('utf-8'))
            await writer.drain()
        user_name: str = checkmessage['name']

        addr = writer.get_extra_info('peername')
        print(f"客户端已连接：{addr}")
        self.clientdict[addr] = False
        while True:
            # 获取音频流并转化为数组
            try:
                data = await asyncio.wait_for(reader.readexactly(self.samples_per_read*8), timeout=1)
            except asyncio.IncompleteReadError:
                break
            except asyncio.TimeoutError:
                break
            sample = np.frombuffer(data, dtype=np.float32).astype(np.float64)
            # 语音识别
            self.stream.accept_waveform(self.sample_rate, sample)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            # 将检测到的结果发送给客户端
            result: str = self.recognizer.get_result(self.stream)
            if result:
                if not self.clientdict[addr]:
                    self.clientdict[addr] = True
                    await self.results_queue.put(ASRActivated(self))
                writer.write((result + "\n").encode('utf-8'))     
            await writer.drain()
            # 如果识别完毕则发出
            if self.recognizer.is_endpoint(self.stream):
                if result:
                    await self.results_queue.put(ASRMessage(self, user_name, result))
                self.recognizer.reset(self.stream)
                self.clientdict[addr] = False
        print(f"客户端 {addr} 已断开连接")
        writer.close()
        await writer.wait_closed()
        del self.clientdict[addr]
        return None
