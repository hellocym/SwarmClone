import socket
from . import config
import json

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((config.PANEL_HOST, config.PANEL_TO_TTS))
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
                print(data['token'], end='', flush=True)
        sock.close()
