from .module_manager import modules
from .controller import Controller
from .modules import *
from .constants import *
from .tts_cosyvoice import TTSCosyvoice
from .frontend_socket import FrontendSocket
from .llm_transformers import LLMTransformers
from .llm_api import LLMOpenAI
from .bilibili_chat import BiliBiliChat
from .asr import ASRSherpa
from .plugins import *
from .ncatbot_modules import *

if __name__ == "__main__":
    print(f"{modules=}")
    controller = Controller(config=Config())
    controller.run()
    pass