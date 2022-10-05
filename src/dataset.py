from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import json


preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FoodOrNotDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        record = self.data[index]
        label = torch.tensor(np.float32(record["label"]))
        img = Image.open(record["path"])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def get_dataloader(data_list_path, batch_size, mode):
    with open(data_list_path, "r") as file:
        data_list = json.load(file)

    if mode == 'train':
        train_list = data_list['train']
        val_list = data_list['val']

        train_dataset = FoodOrNotDataset(train_list, preprocess)
        val_dataset = FoodOrNotDataset(val_list, preprocess)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size)
        return train_loader, val_loader
    else:
        eval_list = data_list['eval']
        eval_dataset = FoodOrNotDataset(eval_list, preprocess)
        return DataLoader(eval_dataset, batch_size)
