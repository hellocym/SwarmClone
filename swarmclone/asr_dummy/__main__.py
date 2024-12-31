import socket
from . import config
import json

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((config.PANEL_HOST, config.PANEL_FROM_ASR))
        while True:
            try:
                message = input("Enter message: ")
            except EOFError:
                break
            data = {
                "from": "ASR",
                "name": "Developer A",
                "text": message
            }
            sock.sendall(json.dumps(data).encode())
