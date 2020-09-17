import torch
import tfrecord
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler


class T5Dataset(Dataset):
    def __init__(self, tfrecord_path: str):
        self.examples = []
        self.count = 0
        records_generator = tfrecord.tfrecord_loader(tfrecord_path, None, {
            "text": "byte",
            "content-length": "byte",
            "content-type": "byte",
            "timestamp": "byte",
            "url": "byte"})
        for record in records_generator:
            self.examples.append(record)
            self.count += 1

    def __len__(self):
        return self.count
    
    def __getitem__(self, i):
        return self.examples[i]


class T5DataCollator:
    def __call__(self, examples):
        return {}


if __name__ == "__main__":
    t5_dataset = T5Dataset(tfrecord_path="./c4-train.tfrecord-00000-of-01024")
    t5_data_collator = T5DataCollator()
    train_sampler = RandomSampler(t5_dataset)
    t5_dataloader = DataLoader(
        t5_dataset,
        batch_size=2,
        sampler=train_sampler,
        collate_fn=t5_data_collator,
    )