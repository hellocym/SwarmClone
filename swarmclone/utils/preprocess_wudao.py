from . import config
from tokenizers import Tokenizer # type: ignore
import numpy as np
from tqdm import tqdm
import ijson # type: ignore

def preprocess_wudao(text_path: str, bin_path: str, tokenizer: Tokenizer):
    with open(text_path, 'r', encoding='utf-8') as f, open(bin_path, 'wb') as f_bin:
        for item in tqdm(ijson.items(f, 'item')):
            dataType = item['dataType']
            title = item['title']
            content = item['content']
            ids = tokenizer.encode(f"【{dataType}】{title}：{content}").ids + [config.SPECIAL_TOKENS["<eos>"]]
            np.array(ids, dtype=np.uint16).tofile(f_bin)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python -m swarmclone.utils.preprocess_wudao <encoder_path> <text_path> <bin_path>')
        exit(1)
    encoder_path = sys.argv[1]
    text_path = sys.argv[2]
    bin_path = sys.argv[3]
    tokenizer = Tokenizer.from_file(encoder_path)
    preprocess_wudao(text_path, bin_path, tokenizer)