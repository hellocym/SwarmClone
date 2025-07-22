# By: Claude Code powered by Kimi K2
import asyncio
import os
import time
import json
import hashlib
from typing import Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .constants import *
from .utils import *
from .modules import *

available_devices = get_devices()

# OpenAI API compatible models
class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant, tool")
    content: str | None = Field(None, description="消息内容")
    name: str | None = Field(None, description="消息发送者名称")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="工具调用列表")
    tool_call_id: str | None = Field(None, description="工具调用ID")

class FunctionDefinition(BaseModel):
    name: str = Field(..., description="函数名称")
    description: str = Field(..., description="函数描述")
    parameters: dict[str, Any] = Field(..., description="函数参数JSON Schema")

class ToolDefinition(BaseModel):
    type: str = Field(default="function", description="工具类型")
    function: FunctionDefinition = Field(..., description="函数定义")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: list[ChatCompletionMessage] = Field(..., description="对话消息列表")
    tools: list[ToolDefinition] | None = Field(None, description="可用工具列表")
    tool_choice: str | dict[str, Any] | None = Field(None, description="工具选择策略")
    temperature: float | None = Field(0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float | None = Field(1.0, ge=0.0, le=1.0, description="核采样阈值")
    max_tokens: int | None = Field(None, ge=1, description="最大生成token数")
    stream: bool | None = Field(False, description="是否流式响应")
    stop: str | list[str] | None = Field(None, description="停止序列")
    presence_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="频率惩罚")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]

@dataclass
class LocalLLMConfig(ModuleConfig):
    model_path: str = field(default="~/.swarmclone/local_models", metadata={
        "required": False,
        "desc": "模型路径"
    })
    model_id: str = field(default="", metadata={
        "required": True,
        "desc": "模型id"
    })
    model_source: str = field(default="modelscope", metadata={
        "required": False,
        "desc": "语言模型来源，仅支持huggingface或modelscope",
        "options": [
            {"key": "Huggingface🤗", "value": "huggingface"},
            {"key": "ModelScope", "value": "modelscope"}
        ],
        "selection": True
    })
    stop_string: str = field(default="\n\n\n", metadata={
        "required": False,
        "desc": "模型输出停止符"
    })
    temperature: float = field(default=0.5, metadata={
        "required": False,
        "desc": "模型温度",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1
    })
    device: str = field(default=[*available_devices.keys()][0], metadata={
        "required": False,
        "desc": "模型运行设备",
        "selection": True,
        "options": [
            {"key": v, "value": k} for k, v in available_devices.items()
        ]
    })
    host: str = field(default="127.0.0.1", metadata={
        "required": False,
        "desc": "服务器监听地址"
    })
    port: int = field(default=9000, metadata={
        "required": False,
        "desc": "服务器端口",
        "min": 1024,
        "max": 65535
    })

class LocalLLM(ModuleBase):
    role: ModuleRoles = ModuleRoles.PLUGIN
    config_class = LocalLLMConfig
    config: config_class
    
    def __init__(self, config: config_class | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self.app = None
        self.server_task = None
        
        abs_model_dir_path = os.path.expanduser(self.config.model_path)
        abs_model_path = os.path.join(abs_model_dir_path, hashlib.md5(self.config.model_id.encode()).hexdigest())
        tries = 0
        while True:
            try:
                print(f"正在从{abs_model_path}加载语言模型……")
                model = AutoModelForCausalLM.from_pretrained(
                    abs_model_path,
                    torch_dtype="auto",
                    trust_remote_code=True
                ).to(self.config.device).bfloat16()
                tokenizer = AutoTokenizer.from_pretrained(
                    abs_model_path,
                    padding_side="left",
                    trust_remote_code=True
                )
                self.model = model
                self.tokenizer = tokenizer
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                break
            except Exception as e:
                tries += 1
                if tries > 5:
                    raise e
                download_model(self.config.model_id, self.config.model_source, abs_model_path)

    def _create_prompt_with_template(self, messages: list[ChatCompletionMessage], tools: list[ToolDefinition] | None = None) -> str:
        """使用模型的chat template生成提示，如果模型不支持tool calling则忽略工具"""
        # 将消息转换为字典格式
        conversation = []
        for msg in messages:
            message_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content or ""
            }
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_calls and msg.role == "assistant":
                message_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id and msg.role == "tool":
                message_dict["tool_call_id"] = msg.tool_call_id
            conversation.append(message_dict)
        
        # 检查模型是否支持tools参数
        supports_tools = False
        try:
            # 测试tokenizer是否支持tools参数
            test_tools = [{"type": "function", "function": {"name": "test", "description": "test", "parameters": {}}}]
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}], 
                tools=test_tools, 
                add_generation_prompt=True,
                tokenize=False
            )
            supports_tools = True
        except Exception:
            supports_tools = False
        
        # 根据支持情况决定是否使用tools
        if tools and supports_tools:
            tools_dict = []
            for tool in tools:
                tools_dict.append({
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    }
                })
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tools=tools_dict,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            # 模型不支持tools，忽略所有工具参数
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
        
        return prompt

    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))

    async def _generate_response(
        self, 
        messages: list[ChatCompletionMessage], 
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        tools: list[ToolDefinition] | None = None
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """生成模型响应，返回(内容,工具调用列表)"""
        prompt = self._create_prompt_with_template(messages, tools)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)
        
        max_new_tokens = max_tokens or 512
        stop_strings = []
        if stop:
            if isinstance(stop, str):
                stop_strings = [stop]
            else:
                stop_strings = stop
        if self.config.stop_string:
            stop_strings.append(self.config.stop_string)

        # 使用线程池执行阻塞操作
        def _generate_sync():
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            # 处理停止符
            for stop_str in stop_strings:
                if stop_str in response:
                    response = response[:response.find(stop_str)]
            
            return response.strip()

        # 在线程池中执行阻塞调用
        response = await asyncio.get_event_loop().run_in_executor(None, _generate_sync)
        
        # 由于使用chat template，工具调用由模型直接处理
        tool_calls = None
        
        return response.strip(), tool_calls

    async def _generate_stream(
        self, 
        messages: list[ChatCompletionMessage], 
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        tools: list[ToolDefinition] | None = None
    ):
        """流式生成响应"""
        prompt = self._create_prompt_with_template(messages, tools)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)
        
        max_new_tokens = max_tokens or 512
        stop_strings = []
        if stop:
            if isinstance(stop, str):
                stop_strings = [stop]
            else:
                stop_strings = stop
        if self.config.stop_string:
            stop_strings.append(self.config.stop_string)

        # 使用transformers流式API
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 准备生成参数
        generation_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "tokenizer": self.tokenizer
        }

        # 如果有停止字符串，添加到生成参数
        if stop_strings:
            generation_kwargs["stop_strings"] = stop_strings

        def _generate_with_streamer():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        # 在后台线程启动生成
        generation_thread = Thread(target=_generate_with_streamer)
        generation_thread.start()

        # 获取流式输出
        for token in streamer:
            yield token

    def _create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        security = HTTPBearer()

        async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """占位式API Key鉴权：接受任意API Key字符串"""
            # 占位式鉴权，接受任意非空字符串
            if not credentials.credentials or not credentials.credentials.strip():
                raise HTTPException(status_code=401, detail="API Key不能为空")
            # 这里可以添加日志记录，但始终返回True
            return True

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            print(f"Local LLM server starting on {self.config.host}:{self.config.port}")
            yield
            print("Local LLM server shutting down")

        app = FastAPI(
            title="Local LLM API",
            description="OpenAI Chat Completion API compatible local LLM server",
            version="1.0.0",
            lifespan=lifespan
        )

        # 添加CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            return {"message": "Local LLM API Server", "model": self.config.model_id}

        @app.get("/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local"
                    }
                ]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest, auth: bool = Depends(verify_api_key)):
            try:
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(request),
                        media_type="text/event-stream"
                    )
                else:
                    return await self._sync_response(request)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health_check(auth: bool = Depends(verify_api_key)):
            return {"status": "healthy", "model_loaded": True}

        return app

    async def _sync_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """同步响应处理"""
        response_text, tool_calls = await self._generate_response(
            messages=request.messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            stop=request.stop,
            tools=request.tools
        )

        prompt_tokens = sum(self._count_tokens(msg.content or "") for msg in request.messages)
        completion_tokens = self._count_tokens(response_text or "")

        # 确定完成原因
        finish_reason = "tool_calls" if tool_calls else "stop"

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", 
                        content=response_text or None, 
                        name=None, 
                        tool_calls=tool_calls, 
                        tool_call_id=None
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    async def _stream_response(self, request: ChatCompletionRequest):
        """流式响应处理"""
        response_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # 发送开始消息
        start_data = {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
        }
        yield f"data: {json.dumps(start_data)}\n\n"

        # 流式生成响应
        response_text = ""
        async for chunk in self._generate_stream(
            messages=request.messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            stop=request.stop,
            tools=request.tools
        ):
            response_text += chunk
            if chunk.strip():  # 只发送非空片段
                chunk_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': request.model,
                    'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        # 发送结束消息
        finish_reason = "stop"
        end_data = {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]
        }
        yield f"data: {json.dumps(end_data)}\n\n"
        yield "data: [DONE]\n\n"

    async def start_server(self):
        """启动服务器"""
        if self.server_task and not self.server_task.done():
            print("服务器已在运行中")
            return

        self.app = self._create_app()
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        
        # 在当前事件循环中运行服务器
        self.server_task = asyncio.create_task(server.serve())
        print(f"Local LLM server started at http://{self.config.host}:{self.config.port}")

    async def stop_server(self):
        """停止服务器"""
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            print("Local LLM server stopped")

    async def run(self):
        """模块主运行函数"""
        await self.start_server()
        # 保持模块运行
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self.stop_server()
            raise
