"""
主控——主控端的核心
"""
import asyncio
from typing import Any
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse

from .modules import *
from .constants import *
from .module_manager import module_classes

from . import __version__

class Controller:
    def __init__(self, config_path: str | None = None):
        self.clear_modules()
        self.app: FastAPI = FastAPI(title="Zhiluo Controller")
        self.register_routes()
        self.module_tasks: list[asyncio.Task[Any]] = []
        self.handler_tasks: list[asyncio.Task[Any]] = []
        self.agent: ModuleBase = ControllerDummy()
    
    def load_config_from_dict(self, config: dict[str, dict[str, dict[str, Any]]]):
        self.clear_modules()
        for role_value, modules in config.items():
            role_classes = module_classes[ModuleRoles(role_value)]
            for name, module_config in modules.items():
                module_class = role_classes[name]
                module = module_class(**module_config)
                self.add_module(module)

    def add_module(self, module: ModuleBase):
        """
        添加模块
        module: 模块
        """
        match module.role:
            case ModuleRoles.LLM:
                if len(self.modules[module.role]) > 0:
                    raise RuntimeError("只能注册一个LLM模块")
            case ModuleRoles.UNSPECIFIED:
                raise ValueError("请指定模块类型")
            case _:
                pass
        assert module.role in self.modules, "不明的模块类型"
        self.modules[module.role].append(module)
    
    def clear_modules(self):
        self.modules: dict[ModuleRoles, list[ModuleBase]] = {
            role: [] for role in ModuleRoles if role not in [ModuleRoles.UNSPECIFIED, ModuleRoles.CONTROLLER]
        }

    def register_routes(self):
        """ 注册FastAPI路由
        
        /:      根路由
        /api:   API路由
        /assets: 静态资源路由
        /api/get_version: 获取版本信息
        """
        self.app.mount("/assets", StaticFiles(directory="panel/dist/assets"), name="assets")
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(open("panel/dist/index.html").read())

        @self.app.post("/api")
        async def api(request: Request):
            try:
                data = await request.json()
            except Exception:
                return JSONResponse(
                    {"error": "Invalid JSON"},
                    status_code=400
                )
            
            if "module" not in data:
                return JSONResponse(
                    {"error": "Missing module field"},
                    status_code=400
                )

            module = data.get("module")
            
            if module == ModuleRoles.ASR.value:
                await self.handle_message(ASRActivated(self.agent))
                message = ASRMessage(
                    source=self.agent,
                    speaker_name=data.get("speaker_name"),
                    message=data.get("content")
                )
                await self.handle_message(message)
            
            return {"status": "OK"}
        
        @self.app.get("/api/get_version")
        async def get_version():
            response = JSONResponse({"version": __version__})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response
    
    async def handle_message(self, message: Message):
        for destination in message.destinations:
            for module_destination in self.modules[destination]:
                await module_destination.task_queue.put(message)
    
    def start_modules(self):
        loop = asyncio.get_event_loop()
        for (module_role, modules) in self.modules.items():
            for i, module in enumerate(filter(lambda x: not x.running, modules)):
                self.module_tasks.append(loop.create_task(module.run(), name=repr(module)))
                self.handler_tasks.append(loop.create_task(self.handle_module(module), name=f"{module_role} handler"))
                print(f"{module}已启动（{i + 1}/{len(modules)}）")
                module.running = True
            if len(modules) > 0:
                print(f"{module_role.value}模块已启动")

    def stop_modules(self):
        for task in self.module_tasks:
            print(f"停止{task.get_name()}模块任务")
            task.cancel()
        for task in self.handler_tasks:
            print(f"停止{task.get_name()}模块任务")
            task.cancel()
        for _role, modules in self.modules.items():
            for module in modules:
                module.running = False
        self.module_tasks.clear()
        self.handler_tasks.clear()

    def run(self):
        self.start_modules()
        
        uvcorn_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8000,
            loop="asyncio"
        )
        server = uvicorn.Server(uvcorn_config)
        loop = asyncio.get_event_loop()
        server_task = loop.create_task(server.serve())
        try:
            loop.run_until_complete(server_task)
        except KeyboardInterrupt:
            self.stop_modules()
            server_task.cancel()
        finally:
            loop.run_until_complete(server.shutdown())
    
    async def handle_module(self, module: ModuleBase):
        while True:
            result: Message = await module.results_queue.get()
            await self.handle_message(result)
    
    def save_as_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            role.value: {module.name: asdict(module.config) for module in modules} for role, modules in self.modules.items()
        }
