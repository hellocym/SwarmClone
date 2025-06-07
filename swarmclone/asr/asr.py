import asyncio
import numpy as np
import json
from typing import Any


from .sherpa_asr import create_recognizer
from .sherpa_vad import create_detector
from ..config import Config
from ..modules import ModuleRoles, ModuleBase
from ..messages import Message, ASRMessage, ASRActivated

class ASRSherpa(ModuleBase):
    def __init__(self, config: Config):
        super().__init__(ModuleRoles.ASR, "ASRsherpa", config)
        self.vad = create_detector(config.asr.sherpa)
        self.recognizer = create_recognizer(config.asr.sherpa)
        self.stream = self.recognizer.create_stream()
        self.sample_rate = 16000
        self.samples_per_read = int(0.1 * self.sample_rate)
        self.userdb = config.asr.userdb
        self.clientdict = {}
        self.speech_started = False
        self.server = None

    async def run(self):
        assert isinstance((host := self.config.panel.server.host), str)
        assert isinstance((port := self.config.asr.port), int)
        self.server = await asyncio.start_server(self.handle_client, host, port)
        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
        try:
            check = await reader.read(1024)
            checkmessage = json.loads(check.decode(), object_hook=dict[str, Any])
        except(UnicodeDecodeError):
            print("不是可接受的鉴权信息")
            return
        try:
            password = getattr(self.userdb, checkmessage['name'])
        except(AttributeError):
            writer.write('WRUSR\n'.encode('utf-8'))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        if not checkmessage['passwd'] == password:
            writer.write('WRPWD\n'.encode('utf-8'))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        else:
            writer.write('OK\n'.encode('utf-8'))
            await writer.drain()
        user_name: str = checkmessage['name']

        addr = writer.get_extra_info('peername')
        print(f"客户端已连接：{addr}")
        self.clientdict[addr[1]] = writer
        while True:
            to_remove = []
            # 获取音频流并转化为数组
            data = await reader.read(self.samples_per_read*8)
            if not data:
                break
            sample = np.frombuffer(data, dtype=np.float32).astype(np.float64)
            # 语音活动检测
            self.vad.accept_waveform(sample)
            if self.vad.is_speech_detected():
                if not self.speech_started:
                    await self.results_queue.put(ASRActivated(self))
                    self.speech_started = True
            else:
                self.speech_started = False
            # 语音识别
            self.stream.accept_waveform(self.sample_rate, sample)
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            # 将检测到的结果发送给客户端
            result: str = self.recognizer.get_result(self.stream)
            writer.write((result + "\n").encode('utf-8'))
            await writer.drain()
            # 如果识别完毕则发出
            if self.recognizer.is_endpoint(self.stream):
                if result:
                    await self.results_queue.put(ASRMessage(self, user_name, result))
                self.recognizer.reset(self.stream)
        print(f"客户端 {addr} 已断开连接")
        writer.close()
        await writer.wait_closed()
        to_remove.append(addr)
        for addr in to_remove:
            del self.clientdict[addr[1]]
        return

    async def process_task(self, task: Message | None) -> Message | None:
    # 不应被调用
        raise NotImplementedError
