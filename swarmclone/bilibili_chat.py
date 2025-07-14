from dataclasses import dataclass, field
from .modules import *
from .messages import *

@dataclass
class BiliBiliChatConfig(ModuleConfig):
    """live_room_id: int，目标B站直播间ID
sessdata、bili_jct、buvid3、dedeuserid、ac_time_value: str，见https://nemo2011.github.io/bilibili-api/#/get-credential，可选
    """
    live_room_id: int = field(default=0)
    sessdata: str = field(default="")
    bili_jct: str = field(default="")
    buvid3: str = field(default="")
    dedeuserid: str = field(default="")
    ac_time_value: str = field(default="")

class BiliBiliChat(ModuleBase):
    role: ModuleRoles = ModuleRoles.CHAT
    config_class = BiliBiliChatConfig
    def __init__(self, config: BiliBiliChatConfig | None = None, **kwargs):
        super().__init__()
        self.config = self.config_class(**kwargs) if config is None else config
        try:
            from bilibili_api import live, Credential
        except ImportError:
            raise ImportError("请安装bilibili-api-python")
        self.config: BiliBiliChatConfig = BiliBiliChatConfig(**kwargs)
        self.credential: Credential = Credential(
            sessdata=self.config.sessdata or None,
            bili_jct=self.config.bili_jct or None,
            buvid3=self.config.buvid3 or None,
            dedeuserid=self.config.dedeuserid or None,
            ac_time_value=self.config.ac_time_value or None,
        )
        self.room: live.LiveDanmaku = live.LiveDanmaku(self.config.live_room_id, credential=self.credential)
        self.register_chat()
    
    def register_chat(self):
        @self.room.on('DANMU_MSG')
        async def on_danmaku(event: dict[str, Any]):
            # 收到弹幕
            print(f"{(user := event['data']['info'][2][1])}: {(msg := event['data']['info'][1])}")
            await self.results_queue.put(
                ChatMessage(
                    source=self,
                    user=user,
                    content=msg,
                )
            )

    async def run(self):
        await self.room.connect()
