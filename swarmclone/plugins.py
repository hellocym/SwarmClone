from dataclasses import field
from typing import Any
from .constants import *
from .messages import *
from .modules import *
from time import time


class ScheduledPlaylistConfig(ModuleConfig):
    playlist: dict[str, dict[str, Any]] = field(default_factory=dict)

class ScheduledPlaylist(ModuleBase):
    role: ModuleRoles = ModuleRoles.PLUGIN
    config_class = ScheduledPlaylistConfig
    def __init__(self, config: ScheduledPlaylistConfig | None = None, **kwargs):
        super().__init__()
        self.config = self.config_class(**kwargs) if config is None else config
        self.playlist = self.config.playlist
    
    async def process_task(self, task: Message | None) -> Message | None:
        for song_id, song_info in self.playlist.items():
            if time() >= song_info["start_time"]:
                del self.playlist[song_id]
                return SongInfo(self, song_id, song_info["file_name"], song_info["subtitle"])
