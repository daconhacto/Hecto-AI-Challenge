import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import collections.abc
import albumentations as A


# --- InitialCustomImageDataset (초기 데이터 로드용) ---
class InitialCustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {root_dir}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((img_path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
        
class FoldSpecificDataset(Dataset):
    def __init__(self, samples_list, image_size, transform=None, is_train=True):
        self.samples_list = samples_list
        self.transform = transform
        self.is_train = is_train
        self.image_size = image_size
        self.is_albu_transform = (isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose))

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        img_path, label = self.samples_list[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            dummy_tensor = torch.zeros((3, self.image_size, self.image_size))
            return (dummy_tensor, 0) if self.is_train else (dummy_tensor, 0, img_path)
        
        # transform이 Albumentations인지 torchvision인지 구분
        if self.is_albu_transform:
            image = np.array(image)
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        return (image, label) if self.is_train else (image, label, img_path)


# --- 테스트 데이터셋 (inf.py에서도 사용) ---
class TestCustomImageDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.image_size = image_size
        self.is_albu_transform = (isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose))
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            print(f"Warning: Test directory {root_dir} not found or is not a directory.")
            return
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root_dir, fname)
                self.samples.append((img_path,))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # print(f"Warning: File not found {img_path}, returning a dummy image.")
            return torch.zeros((3, self.image_size, self.image_size))
        if self.transform:
            if self.is_albu_transform:
                image = np.array(image)
                image = self.transform(image=image)['image']
            else:
                image = self.transform(image)
        return image