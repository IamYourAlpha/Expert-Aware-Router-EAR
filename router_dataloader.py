''' Returns MLC dataloader for the router network'''
import torch
import os
import time
import json
import copy
import numpy as np
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms



class RouterDataLoader(Dataset):
    def __init__(self, base_path, train_or_test, confusing_classes, transform=None):
        self.img_path = os.path.join(base_path, train_or_test)
        self.transform = transform
        # Get list of all the images 
        self.list_of_images = os.listdir(self.img_path)
        # Get the list. for all the confusing class pairs.
        self.confusing_classes_pairs = confusing_classes

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.list_of_images[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_str = self.list_of_images[index].split('_')
        label_int = int(label_str[0])
        mlc_label = [] 
        for confusing_pair in self.confusing_classes_pairs:
            if (label_int in confusing_pair):
                mlc_label.append(1)
            else:
                mlc_label.append(0)
        mlc_label = np.array(mlc_label)
        
        mlc_label = torch.from_numpy(mlc_label)
        mlc_label = mlc_label.float()
        #label_np = np.array(label_int)
        #label = torch.from_numpy(label_np)
        return img, mlc_label, label_int

    def __len__(self):
        return len(self.list_of_images)



# dataloc = "F:/Research/PHD_AIZU/tiny_ai/ear/data/c100_combined"
# train_or_test = "train"
# batch_size = 4
# confusing_classes = [[35, 98], [55, 72], [47, 52], [11, 35], [11, 46], [70, 92], [13, 81], [47, 96], [2, 35], [81, 90]]
# tranformation = transforms.Compose([
#                     transforms.ToTensor()])
# loader = RouterDataLoader(dataloc, train_or_test, confusing_classes, tranformation)
# loader = DataLoader(loader, batch_size, shuffle=True)
# batch, labels = next(iter(loader))
# print (labels)
