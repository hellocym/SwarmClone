from modelscope import snapshot_download as modelscope_snapshot_download
from huggingface_hub import snapshot_download as huggingface_snapshot_download
import re

def download_model(model_id: str, model_source: str, local_dir: str):
    match model_source:
        case "modelscope":
            modelscope_snapshot_download(model_id, local_dir=local_dir, repo_type="model")
        case "huggingface":
            huggingface_snapshot_download(model_id, local_dir=local_dir, repo_type="model")
        case x if x.startswith("openai+"):
            raise ValueError((
                f"OpenAI API模型不能被下载。"
                "如果你在使用`LLMTransformers`（默认选项）时遇到了这个问题，"
                "请转而使用`LLMOpenAI`。"
            ))
        case _:
            raise ValueError(f"Invalid model source: {model_source}")

def split_text(s: str, separators: str="。？！～.?!~\n\r") -> list[str]: # By DeepSeek
    # 构建正则表达式模式
    separators_class = ''.join(map(re.escape, separators))
    pattern = re.compile(rf'([{separators_class}]+)')
    
    # 分割并处理结果
    parts = pattern.split(s)
    result = []
    
    # 合并文本与分隔符（成对处理）
    for text, delim in zip(parts[::2], parts[1::2]):
        if (cleaned := (text + delim).lstrip()):
            result.append(cleaned)
    
    # 处理未尾未配对内容（保留后置空格）
    if len(parts) % 2:
        if (last_cleaned := parts[-1].lstrip()):
            result.append(last_cleaned)
    
    return result

