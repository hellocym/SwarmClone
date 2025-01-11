import os
import sys
import json
import socket
import shutil
import tempfile
import warnings
import threading

from time import time
from queue import Queue
from typing import Optional

import playsound
import torchaudio

from . import config, tts_config
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
        playsound.playsound(os.path.abspath(fname))
        os.remove(fname)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
    warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only=False.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_norm.*")
    
    temp_dir = tempfile.gettempdir()
    try:
        model_path = os.path.expanduser(os.path.join(tts_config.MODELPATH, tts_config.MODEL))
        cosyvoice = CosyVoice(model_path, fp16=tts_config.FLOAT16)
    except Exception as e:
        err_msg = str(e).lower()
        if ("file" in err_msg) and ("doesn't" in err_msg) and ("exist" in err_msg):
            catch = input(" * S.C. CosyVoice TTS 发生了错误，这可能是由于模型下载不完全导致的，是否清理缓存TTS模型？[y/n] ")
            if catch.strip().lower() == "y":
                shutil.rmtree(os.path.expanduser(tts_config.MODELPATH), ignore_errors=True)
                print(" * 清理完成，请重新运行该模块。")
                sys.exit(0)
            else:
                raise
        else:
            raise
        
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
                fname = os.path.join(temp_dir, f"voice{time()}.mp3")
                torchaudio.save(fname, outputs, 22050)
                q_fname.put(fname)
        sock.close()
        q_fname.put(None)
        get_data_thread.join()
        play_sound_thread.join()
    