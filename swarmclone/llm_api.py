import asyncio
import os
import torch
import openai
from dataclasses import dataclass, field
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from .modules import *
from .messages import *
from .utils import *

@dataclass
class LLMOpenAIConfig(LLMBaseConfig):
    classifier_model_path: str = field(default="~/.swarmclone/llm/EmotionClassification/SWCBiLSTM", metadata={
        "required": False,
        "desc": "情感分类模型路径"
    })
    classifier_model_id: str = field(default="MomoiaMoia/SWCBiLSTM", metadata={
        "required": False,
        "desc": "情感分类模型id"
    })
    classifier_model_source: str = field(default="modelscope", metadata={
        "required": False,
        "desc": "情感分类模型来源，仅支持huggingface或modelscope"
    })
    model_id: str = field(default="", metadata={
        "required": True,
        "desc": "模型id"
    })
    model_url: str = field(default="", metadata={
        "required": True,
        "desc": "模型api网址"
    })
    api_key: str = field(default="", metadata={
        "required": True,
        "desc": "api key"
    })
    temperature: float = field(default=0.7, metadata={
        "required": False,
        "desc": "模型温度",
        "selection": False,
        "options": [
            {"key": "0.7", "value": 0.7},
            {"key": "0.9", "value": 0.9},
            {"key": "1.0", "value": 1.0}
        ]
    })

class LLMOpenAI(LLMBase):
    role: ModuleRoles = ModuleRoles.LLM
    config_class = LLMOpenAIConfig
    def __init__(self, config: LLMOpenAIConfig | None = None, **kwargs):
        super().__init__()
        self.config = self.config_class(**kwargs) if config is None else config
        abs_classifier_path = os.path.expanduser(self.config.classifier_model_path)
        successful = False
        while not successful: # 加载情感分类模型
            try:
                print(f"正在从{abs_classifier_path}加载情感分类模型……")
                classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    abs_classifier_path,
                    torch_dtype="auto",
                    trust_remote_code=True
                ).to("cpu")
                classifier_tokenizer = AutoTokenizer.from_pretrained(
                    abs_classifier_path,
                    padding_side="left",
                    trust_remote_code=True
                )
                successful = True
                self.classifier_model = classifier_model
                self.classifier_tokenizer = classifier_tokenizer
            except Exception:
                download_model(
                    self.config.classifier_model_id,
                    self.config.classifier_model_source,
                    abs_classifier_path
                )
        
        self.model_id = self.config.model_id
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.model_url
        )
        self.temperature = self.config.temperature
    
    @torch.no_grad()
    async def get_emotion(self, text: str) -> dict[str, float]:
        print(text)
        labels = ['neutral', 'like', 'sad', 'disgust', 'anger', 'happy']
        ids = self.classifier_tokenizer([text], return_tensors="pt")['input_ids']
        probs = (
            (await asyncio.to_thread(self.classifier_model, input_ids=ids))
            .logits
            .softmax(dim=-1)
            .squeeze()
        )
        return dict(zip(labels, probs.tolist()))
    
    def dict2message(self, message: dict[str, str]):
        match message:
            case {'role': 'user', 'content': content}:
                return ChatCompletionUserMessageParam(role="user", content=content)
            case {'role': 'assistant', 'content': content}:
                return ChatCompletionAssistantMessageParam(role="assistant", content=content)
            case {'role': 'system', 'content': content}:
                return ChatCompletionSystemMessageParam(role="system", content=content)
            case _:
                raise ValueError(f"Invalid message: {message}")
    
    async def iter_sentences_emotions(self):
        generating_sentence = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[self.dict2message(message) for message in self.history],
                stream=True,
                temperature=self.temperature
            )
            async for delta in response:
                t = delta.choices[0].delta.content
                generating_sentence += (t or "")
                self.generated_text += (t or "")
                if (sentences := split_text(generating_sentence))[:-1]:
                    for sentence in sentences:
                        if (sentence := sentence.strip()):
                            yield sentence, await self.get_emotion(sentence)
                    generating_sentence = sentences[-1]
        finally: # 被中断或者生成完毕
            yield generating_sentence, await self.get_emotion(generating_sentence)
