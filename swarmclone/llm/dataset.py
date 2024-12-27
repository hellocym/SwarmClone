from typing import Optional
from torch.utils.data import Dataset
import torch
import numpy as np
from . import config

class PreTrainDataset(Dataset):
    def __init__(self, data_path: str, max_lenth: int, unused_indexes: Optional[list[int]] = None):
        self.data_path = data_path
        self.max_lenth = max_lenth
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.n_lines = data.shape[0] // (max_lenth + 1)
        self.data = data[:self.n_lines * (max_lenth + 1)].reshape((self.n_lines, max_lenth + 1))

        if unused_indexes is None:
            self.unused_indexes = list(range(self.n_lines)) # 默认所有行都未使用
        else:
            self.unused_indexes = unused_indexes
        self.used_indexes: list[int] = [] # 已使用的行索引

    def __len__(self):
        return len(self.unused_indexes) # 返回未使用的行数

    def __getitem__(self, index):
        abs_index = self.unused_indexes[index] # 从索引表中取出绝对索引
        line = self.data[abs_index]
        self.used_indexes.append(abs_index) # 记录已使用的行索引
        x = torch.from_numpy(line[:-1])
        y = torch.from_numpy(line[1:])
        return x, y

    def get_unused_indexes(self):
        unused_indexes = self.unused_indexes[:]
        for used in self.used_indexes:
            unused_indexes.remove(used)
        return unused_indexes

def collate_fn(batch):
    x_list, y_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from tokenizers import Tokenizer # type: ignore
    if len(sys.argv) < 3:
        print("Usage: python -m swarmclone.llm.dataset <tokenizer_path> <data_path>")
        exit(1)
    tokenizer_path = sys.argv[1]
    data_path = sys.argv[2]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = PreTrainDataset(data_path, config.MAX_LENGTH,
                            unused_indexes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # 只取前10行作为测试
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    for x, y in dataloader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        for i in range(config.BATCH_SIZE):
            print(tokenizer.decode(x[i].tolist())[:50])
            print(tokenizer.decode(y[i].tolist())[:50])
        try:
            input()
        except EOFError:
            break
    print(dataset.get_unused_indexes())
