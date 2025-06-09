import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import collections.abc
import albumentations as A
from torch.utils.data import Sampler
import random
import torchvision.transforms.functional as TF

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
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.is_albu_transform = (isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose))

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        img_path, label = self.samples_list[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            dummy_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]))
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
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
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
            return torch.zeros((3, self.image_size[0], self.image_size[1]))
        if self.transform:
            if self.is_albu_transform:
                image = np.array(image)
                image = self.transform(image=image)['image']
            else:
                image = self.transform(image)
        return image



class GroupedBatchSampler(Sampler):
    def __init__(self, label_to_indices, class_groups, batch_size, shuffle=True):
        self.label_to_indices = label_to_indices
        self.class_groups = class_groups
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.group_batches = []
        self._prepare_batches()

    def _prepare_batches(self):
        self.group_batches.clear()

        for group_classes in self.class_groups.values():
            group_indices = []
            for cls in group_classes:
                group_indices.extend(self.label_to_indices[cls])

            if self.shuffle:
                random.shuffle(group_indices)

            # 배치 단위로 나누기
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    self.group_batches.append(batch)

        if self.shuffle:
            random.shuffle(self.group_batches)

    def __iter__(self):
        if self.shuffle:
            self._prepare_batches()
        for batch in self.group_batches:
            yield batch

    def __len__(self):
        return len(self.group_batches)


class TTATestCustomImageDataset(Dataset):
    def __init__(self, root_dir, transform, img_size, tta_times=4):
        """
        Args:
            root_dir (str): 테스트 이미지가 있는 폴더 경로
            transform (Transform): 단일 transform (e.g. train_transform)
            tta_times (int): 동일 이미지에 몇 번 transform을 적용할지
        """
        self.root_dir = root_dir
        self.transform = transform
        self.tta_times = tta_times
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.samples = []
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
            return [torch.zeros((3, self.img_size[0], self.img_size[1])) for _ in range(self.tta_times)]

        # transform이 Albumentations인지 torchvision인지 구분
        if self.is_albu_transform:
            images = [self.transform(image=np.array(image))['image'] for _ in range(self.tta_times)]
        else:
            images = [self.transform(image) for _ in range(self.tta_times)]
        return images  # (tta_times, C, H, W)


import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os

class KnowledgeDistillationDataset(Dataset):
    def __init__(self, samples_list, image_size, transform, teacher_logits_list, teacher_val_scores):
        """
        Args:
            samples_list (List[Tuple[str, int]]): (img_path, label) 리스트
            image_size (Tuple[int, int]): 이미지 크기 (H, W)
            transform: 이미지에 적용할 transform (albumentations 또는 torchvision)
            teacher_logits_list (List[pd.DataFrame]): img_path를 index로, 각 클래스별 logits가 있는 DataFrame 리스트
            teacher_val_scores (List[float]) : 앙상블 teacher model의 val score를 담은 list
        """
        self.samples_list = samples_list
        self.image_size = image_size
        self.transform = transform
        self.is_albu_transform = (isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose))
        self.teacher_model_weights = [x / sum(teacher_val_scores) for x in teacher_val_scores]

        # Step 1: 각 DataFrame에서 'ID'를 index로 설정하고 정렬
        for i in range(len(teacher_logits_list)):
            df = teacher_logits_list[i]
            if 'ID' not in df.columns:
                raise ValueError(f"teacher_logits_list[{i}]에 'ID' column이 없습니다.")
            df = df.set_index('ID')
            df = df.sort_index()
            teacher_logits_list[i] = df

        # Step 2: 공통된 ID만 사용
        common_ids = teacher_logits_list[0].index
        for df in teacher_logits_list[1:]:
            common_ids = common_ids.intersection(df.index)

        # Step 3: 공통된 ID만 남기고 정렬
        for i in range(len(teacher_logits_list)):
            teacher_logits_list[i] = teacher_logits_list[i].loc[common_ids]

        # Step 4: 가중 평균 계산 (DataFrame끼리 계산, 결과도 DataFrame)
        self.teacher_logits_df = sum(w * df for w, df in zip(self.teacher_model_weights, teacher_logits_list))


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        img_path, label = self.samples_list[idx]

        # 이미지 로드
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.fromarray(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))

        # transform이 Albumentations인지 torchvision인지 구분
        if self.is_albu_transform:
            image = np.array(image)
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        # teacher logits
        if img_path in self.teacher_logits_df.index:
            logits = self.teacher_logits_df.loc[img_path].values.astype(np.float32)
        else:
            # 예외 처리: 해당 이미지에 대한 로그잇이 없을 경우 0 벡터 반환
            num_classes = self.teacher_logits_df.shape[1]
            logits = np.zeros(num_classes, dtype=np.float32)

        return image, label, logits
