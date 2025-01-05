from transformers import pipeline, TextIteratorStreamer # type: ignore
import socket
import threading
import json
import queue
from . import config

q: queue.Queue[dict[str, str]] = queue.Queue()

def recv_msg(sock: socket.socket, q: queue.Queue[dict[str, str]]):
    while True:
        try:
            data = json.loads(sock.recv(1024).decode())
        except json.JSONDecodeError:
            pass
        else:
            if data["from"] == "stop":
                q.put({"role": "stop", "content": ""})
                break
            if data["from"] == "ASR":
                q.put({"role": "user", "content": data["text"]})
            else:
                for message in data["list"]:
                    q.put({"role": "user", "content": message["text"]})

if __name__ == '__main__':
    pipe = pipeline(
        'text-generation',
        model='Qwen/Qwen2.5-0.5B-Instruct',
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=1000,
    )
    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
    with (  socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_input,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_output):
        sock_input.connect((config.PANEL_HOST, config.PANEL_TO_LLM))
        sock_output.connect((config.PANEL_HOST, config.PANEL_FROM_LLM))
        t_recv = threading.Thread(target=recv_msg, args=(sock_input, q))
        t_recv.start()
        messages: list[dict[str, str]] = []
        while True:
            message = q.get(True)
            if message["role"] == "stop":
                sock_output.sendall(json.dumps({"from": "stop"}).encode())
                sock_output.close()
                sock_input.close()
                break
            messages.append(message)
            kwargs = {"text_inputs": messages, "streamer": streamer}
            generation_thread = threading.Thread(target=pipe, kwargs=kwargs)
            generation_thread.start()
            generated_text = ""
            for text in streamer:
                data = {
                    "from": "LLM",
                    "token": text,
                    "feelings": {}
                }
                sock_output.sendall(json.dumps(data).encode())
                generated_text += text
            data = {
                "from": "LLM",
                "token": "<eos>",
                "feelings": {}
            } # 表示生成结束
            sock_output.sendall(json.dumps(data).encode())
            messages.append({"role": "assistant", "content": generated_text})
        