from dataclasses import dataclass, field

@dataclass
class Config:
    # Tokenizer参数
    VOCAB_SIZE: int = 32768
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2})
    
    # 模型参数
    MAX_LENGTH: int = 1024
    MODEL_DIM: int = 1024
    NUM_HEADS: int = 12
    NUM_LAYERS: int = 16
    DROPOUT: float = 0.1

    # 训练参数
    BATCH_SIZE: int = 1
    N_BATCHES_PER_STEP: int = 100
    MAX_LEARNING_RATE: float = 1e-4
    MIN_LEARNING_RATE: float = 1e-6
    WARMUP_STEPS: int = 10000
    TOTAL_STEPS: int = 1000000
