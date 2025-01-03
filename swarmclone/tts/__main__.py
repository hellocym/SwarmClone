import socket
from . import config
import json
from queue import Queue
import threading
from typing import Optional
from TTS.tts.configs.xtts_config import XttsConfig # type: ignore
from TTS.tts.models.xtts import Xtts # type: ignore
import soundfile as sf # type: ignore
import playsound # type: ignore

def get_data(sock: socket.socket, q: Queue[Optional[str]]):
    s = ""
    while True:
        msg = sock.recv(1024)
        if not msg:
            break
        try:
            data = json.loads(msg.decode('utf-8'))
        except:
            continue
        if data['from'] == "stop":
            break
        if data['from'] == "LLM":
            token = data['token']
            s += data['token']
            if any(c in s for c in ",，.。!！?？\n\r……()（）[]"):
                q.put(s)
                s = ""
    q.put(None)

if __name__ == '__main__':
    xtts_config = XttsConfig()
    xtts_config.load_json("coqui/XTTS-v2/config.json")
    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(xtts_config, "coqui/XTTS-v2", eval=True)
    model.cuda()

    q: Queue[Optional[str]] = Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((config.PANEL_HOST, config.PANEL_TO_TTS))
        get_data_thread = threading.Thread(target=get_data, args=(sock, q))
        get_data_thread.start()
        while True:
            if not q.empty():
                s = q.get()
                if s is None:
                    break
                print(s)
                outputs = model.synthesize(
                    s,
                    xtts_config,
                    speaker_wav="/data/编曲/声音/人声采样/evil.wav",
                    language="en" if s.encode('utf-8').isascii() else "zh"
                )
                sf.write("/tmp/output.wav", outputs["wav"], 22050)
                playsound.playsound("/tmp/output.wav")
        sock.close()
        get_data_thread.join()
