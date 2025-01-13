import socket
import threading
import json
import queue
import os
from transformers import pipeline, TextIteratorStreamer  # type: ignore
from . import config, qwen2_config

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

def load_model():
    successful = False
    abs_model_path = os.path.expanduser(qwen2_config.MODEL_PATH)
    while not successful:
        try:
            pipe = pipeline(
                'text-generation',
                model=abs_model_path,
                torch_dtype="auto",
                device_map="auto",
                max_new_tokens=1000,
            )
            successful = True
        except:
            choice = input("加载模型失败，是否下载模型？(Y/n)")
            if choice.lower() != "n":
                import huggingface_hub  # type: ignore
                huggingface_hub.snapshot_download(
                    repo_id=qwen2_config.MODEL_ID,
                    repo_type="model",
                    local_dir=abs_model_path,
                    endpoint="https://hf-mirror.com"
                )
    return pipe

def llm_send_to_server(data: dict):
    
    pass

def server_send_to_llm():
    # TODO:从服务器获取数据
    # 返回一个消息字典
    return {"Tom": "hi"}

def start_processing(pipe):
    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
    with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_input,
          socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_output):
        sock_input.connect((config.PANEL_HOST, config.PANEL_TO_LLM))
        sock_output.connect((config.PANEL_HOST, config.PANEL_FROM_LLM))
        t_recv = threading.Thread(target=recv_msg, args=(sock_input, q))
        t_recv.start()
        messages: list[dict[str, str]] = []
        while True:
            message = server_send_to_llm()  # 获取消息
            if message["role"] == "stop":
                llm_send_to_server({"from": "stop"})
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
                llm_send_to_server(data)  # 发送数据
                generated_text += text
            data = {
                "from": "LLM",
                "token": "<eos>",
                "feelings": {}
            }  # 表示生成结束
            llm_send_to_server(data)
            messages.append({"role": "assistant", "content": generated_text})