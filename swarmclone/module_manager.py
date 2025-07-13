from .config import Config
from .constants import *
from .messages import *
import asyncio

class ModuleManager(type):
    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]):
        new_class = super().__new__(cls, name, bases, attrs)
        if name != "ModuleBase":
            print(f"Registering module {name}")
            modules[name] = new_class
        return new_class

modules: dict[str, ModuleManager] = {}

class ModuleBase(metaclass=ModuleManager):
    def __init__(self, module_role: ModuleRoles, name: str, config: Config):
        self.name: str = name
        self.role: ModuleRoles = module_role
        self.task_queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=128)
        self.results_queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=128)
        self.config: Config = config
    
    async def run(self) -> None:
        while True:
            try:
                task = self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                task = None
            result = await self.process_task(task)
            if result is not None:
                await self.results_queue.put(result)
            await asyncio.sleep(0.1)

    def __repr__(self):
        return f"<{self.role} {self.name}>"

    async def process_task(self, task: Message | None) -> Message | None:
        """
        处理任务的方法，每个循环会自动调用
        返回None表示不需要返回结果，返回Message对象则表示需要返回结果，返回的对象会自动放入results_queue中。
        也可以选择手动往results_queue中put结果然后返回None
        """
