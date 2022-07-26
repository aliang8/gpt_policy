from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Dict, List, Any
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(self, hparams: Dict):
        self.hparams = hparams

    def get_sampler(self, indices):
        return None  # use default batch sampler

    def collate_fn(self, data):
        return default_collate(data)
