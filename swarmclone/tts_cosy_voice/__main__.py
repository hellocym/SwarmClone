import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party/Matcha-TTS'))

import json
import socket
import playsound
import threading
import soundfile as sf
import torchaudio
import warnings

from time import time
from . import config
from queue import Queue
from typing import Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

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
            if token == "<eos>":
                q.put(s)
                s = ""
                continue
            s += data['token']
            for sep in ".!?。？！…\n\r":
                if sep in token:
                    splits = s.split(sep)
                    for split in splits[:-1]:
                        if len(split.strip()) > 1:
                            q.put(split + sep)
                    s = splits[-1]
                    break
            if not s.isascii() and len(s) >= 50:
                q.put(s)
                s = ""
    q.put(None)

def play_sound(q_fname: Queue[Optional[str]]):
    while True:
        fname = q_fname.get()
        if fname is None:
            break
        playsound.playsound(fname)
        os.remove(fname)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
    warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm is deprecated.*")
    
    base_dir = os.path.dirname(__file__)
    pretrained_model_path = os.path.join(base_dir, 'pretrained_models', 'CosyVoice-300M-SFT')
    # prompt_wav_path = os.path.join(base_dir, 'zero_shot_prompt.wav')
    
    cosyvoice = CosyVoice(pretrained_model_path)

    q: Queue[Optional[str]] = Queue()
    q_fname: Queue[Optional[str]] = Queue()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((config.PANEL_HOST, config.PANEL_TO_TTS))
        get_data_thread = threading.Thread(target=get_data, args=(sock, q))
        get_data_thread.start()
        play_sound_thread = threading.Thread(target=play_sound, args=(q_fname,))
        play_sound_thread.start()
        while True:
            if not q.empty():
                s = q.get()
                if s is None:
                    break
                if not s or s.isspace():
                    continue
                outputs = list(cosyvoice.inference_sft(s, '中文女', stream=False))[0]["tts_speech"]
                fname = f"/tmp/voice{time()}.wav"
                torchaudio.save(fname, outputs, 22050)
                q_fname.put(fname)
        sock.close()
        q_fname.put(None)
        get_data_thread.join()
        play_sound_thread.join()
    