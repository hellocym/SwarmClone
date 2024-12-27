from torch.utils.data import Dataset
import torch
import numpy as np
from . import config

class PreTrainDataset(Dataset):
    def __init__(self, data_path: str, max_lenth: int):
        self.data_path = data_path
        self.max_lenth = max_lenth
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.n_lines = data.shape[0] // (max_lenth + 1)
        self.data = data[:self.n_lines * (max_lenth + 1)].reshape((self.n_lines, max_lenth + 1))

    def __len__(self):
        return self.n_lines

    def __getitem__(self, index):
        line = self.data[index]
        x = torch.from_numpy(line[:-1])
        y = torch.from_numpy(line[1:])
        return x, y

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
    dataset = PreTrainDataset(data_path, config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    for x, y in dataloader:
        print(x)
        print(y)
        for i in range(config.BATCH_SIZE):
            print(tokenizer.decode(x[i].tolist())[:50])
            print(tokenizer.decode(y[i].tolist())[:50])
        try:
            input()
        except EOFError:
            break
