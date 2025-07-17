"""
主控——主控端的核心
"""
from starlette.responses import JSONResponse


import asyncio
from typing import Any
from dataclasses import asdict, fields, MISSING

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse

from .modules import *
from .constants import *
from .module_manager import module_classes
from .utils import *

from . import __version__

class Controller:
    def __init__(self, config_path: str | None = None):
        self.clear_modules()
        self.app: FastAPI = FastAPI(title="Zhiluo Controller")
        self.register_routes()
        self.module_tasks: list[asyncio.Task[Any]] = []
        self.handler_tasks: list[asyncio.Task[Any]] = []
        self.agent: ModuleBase = ControllerDummy()

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
        /api/get_config: 获取配置信息
        """
        self.app.mount("/assets", StaticFiles(directory="panel/dist/assets"), name="assets")
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(open("panel/dist/index.html").read())
        
        @self.app.get("/api/get_config", response_class=JSONResponse)
        async def get_config() -> JSONResponse:
            """
            [
                {
                    "role_name":【模块角色】,
                    "modules":[
                        {
                            "module_name":【模块名字】
                            "desc":【介绍】,
                            "config":[
                                {
                                    "name":【配置项名字】
                                    "type":【类型，int整数float小数（默认小数点后2位精度）str字符串bool布尔值（是/否）selection选择项】,
                                    "desc":【介绍信息】,
                                    "required":【布尔值，是否必填】,
                                    "default":【默认值】,
                                    "options":【可选项，无论类型均可提供，若非选择项仍提供选项则说明有预设值可选（比如模型可选deepseek-v3/qwen3之类），若为空则为无选项】
                                },...
                            ]
                        },...
                    ]
                },...
            ]
            """
            config: list[Any] = []
            for role, role_module_classes in module_classes.items():
                config.append({"role_name": role.value, "modules": []})
                for module_name, module_class in role_module_classes.items():
                    config[-1]["modules"].append({
                        "module_name": module_name,
                        "desc": module_class.__doc__ or "",
                        "config": []
                    })
                    for field in fields(module_class.config_class): # config_class是被其元类注入的，是dataclass
                        name = field.name
                        default = ""
                        # 将各种类型转换为字符串表示
                        _type: str
                        raw_type = field.type
                        if isinstance(raw_type, type):
                            raw_type = raw_type.__name__
                        if "int" in raw_type and "float" not in raw_type: # 只在一个参数只能是int而不能是float时确定其为int
                            _type = "int" # TODO: if "int" == raw_type好像也行？
                        elif "float" in raw_type:
                            _type = "float"
                        elif "bool" in raw_type:
                            _type = "bool"
                        else:
                            _type = "str"
                        selection = field.metadata.get("selection", False)
                        if selection:
                            _type = "selection" # 如果是选择项则不管类型如何
                        
                        required = field.metadata.get("required", False)

                        desc = field.metadata.get("desc", "")

                        options = field.metadata.get("options", [])

                        if field.default is not MISSING and (default := field.default) is not None:
                            pass
                        elif field.default_factory is not MISSING and (default := field.default_factory()) is not None:
                            pass
                        else: # 无默认值则生成对应类型的空值
                            if _type == "str":
                                default = ""
                            elif _type == "int":
                                default = 0
                            elif _type == "float":
                                default = 0.0
                            elif _type == "bool":
                                default = False
                            elif _type == "selection":
                                default = options[0]["value"]
                        if isinstance(default, str):
                            default = escape_all(default) # 进行转义
                        
                        # 如果有的话，提供最大最小值和步长
                        if "min" in field.metadata:
                            minimum = field.metadata["min"]
                        else:
                            minimum = None
                        if "max" in field.metadata:
                            maximum = field.metadata["max"]
                        else:
                            maximum = None
                        if "step" in field.metadata:
                            step = field.metadata["step"]
                        else:
                            step = None

                        config[-1]["modules"][-1]["config"].append({
                            "name": name,
                            "type": _type,
                            "desc": desc,
                            "required": required,
                            "default": default,
                            "options": options,
                            "min": minimum,
                            "max": maximum,
                            "step": step
                        })
            return JSONResponse(config)
        
        @self.app.post("/api/load_config", response_class=JSONResponse)
        async def load_config(request: Request) -> JSONResponse:
            """
            [
                {
                    "role_name":【模块角色】,
                    "modules":[
                        {
                            "module_name":【模块名字】
                            "config":[
                                {
                                    "name":【配置项名字】,
                                    "value":【配置值】
                                },...
                            ]
                        },...
                    ]
                },...
            ]
            """
            data = await request.json()
            self.clear_modules()
            missing_modules: list[str] = []
            for role in data:
                role_name = role["role_name"]
                role_modules = role["modules"]
                for module in role_modules:
                    module_name = module["module_name"]
                    module_config = module["config"]
                    try:
                        module_class = module_classes[ModuleRoles(role_name)][module_name]
                    except KeyError:
                        missing_modules.append(module_name)
                        continue
                    
                    # 去转义
                    for key, value in module_config.items():
                        if isinstance(value, str):
                            module_config[key] = unescape_all(value)
                    try:
                        module = module_class(**module_config)
                    except Exception as e:
                        return JSONResponse({"error": str(e)}, 500)
                    self.add_module(module)
            if missing_modules:
                return JSONResponse(missing_modules, 404)
            return JSONResponse({"status": "OK"})

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
