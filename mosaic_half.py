# train.py하고 크게 다를 건 없으나 Albumentation을 적용해보기 위해 작성한 .py파일입니다.

import os
import sys
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json # class_names 저장을 위해 추가
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import log_loss
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import collections.abc  # for checking if transform is callable
from augmentations import *

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter Setting
CFG = {
    "ROOT": '/home/sh/hecto/train',
    "WORK_DIR": '/home/sh/hecto/tjrgus5/work_dir/convnext1260_half_image_retraining_with_wrongs_2',
    "START_FROM": '/home/sh/hecto/tjrgus5/work_dir/convnext_mosaic_or_cutmix+mixup_test/best_model_convnext_base.fb_in22k_ft_in1k_384_fold1.pth', # 만약 None이 아닌 .pth파일 경로 입력하면 해당 checkpoint를 load해서 시작

    "CUTMIX": False,
    "MIXUP": False,
    "CUTOUT": False,

    # --- 새로운 Mosaic 관련 설정 ---
    'APPLY_MOSAIC_GROUP_P': 1, # Mosaic 계열(Half 또는 Standard) 증강을 적용할 전체 확률
    'HALF_MOSAIC_ENABLED': True,    # Half-Mosaic 사용 여부 (마스터 스위치)
    'STANDARD_MOSAIC_ENABLED': False,  # Standard (4-cell) Mosaic 사용 여부 (마스터 스위치)
                                     # 기존 HALF_MOSAIC_P, MOSAIC_P는 이 로직에서 사용되지 않음
    # --------------------------------

    'IMG_SIZE': 512,
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-5,
    'SEED' : 42,
    'MODEL_NAME': 'convnext_base.fb_in22k_ft_in1k_384',
    'N_FOLDS': 5,
    'EARLY_STOPPING_PATIENCE': 3,
    'RUN_SINGLE_FOLD': True,
    'TARGET_FOLD': 1
}

# 1) 네가 제공한 혼동 클래스 쌍 리스트
confusion_pairs = [
    ("아반떼_하이브리드_CN7_2021_2023", "아반떼_CN7_2021_2023"),
    ("GLC_클래스_X253_2020_2022", "GLC_클래스_X253_2023"),
    ("K8_2022_2024", "K8_하이브리드_2022_2024"),
    ("트레일블레이저_2023", "트레일블레이저_2021_2022"),
    ("K7_프리미어_하이브리드_2020_2021", "K7_프리미어_2020_2021"),
    ("4시리즈_G22_2021_2023", "4시리즈_G22_2024_2025"),
    ("더_넥스트_스파크_2016_2018", "더_뉴_스파크_2019_2022"),
    ("더_뉴_K5_3세대_2024_2025", "더_뉴_K5_하이브리드_3세대_2023_2025"),
    ("레인지로버_4세대_2014_2017", "레인지로버_4세대_2018_2022"),
    ("5008_2세대_2021_2024", "3008_2세대_2018_2023"),
    ("3008_2세대_2018_2023", "5008_2세대_2018_2019"),
    ("7시리즈_G11_2016_2018", "7시리즈_G11_2019_2022"),
    ("EQE_V295_2022_2024", "EQS_V297_2022_2023"),
    ("K5_3세대_하이브리드_2020_2022", "K5_3세대_2020_2023"),
    ("라브4_5세대_2019_2024", "RAV4_5세대_2019_2024"),
    ("레인지로버_이보크_2세대_2023_2024", "레인지로버_이보크_2세대_2020_2022"),
    ("스팅어_마이스터_2021_2023", "스팅어_2018_2020"),
    ("3시리즈_GT_F34_2014_2021", "4시리즈_F32_2014_2020"),
    ("GLE_클래스_W166_2016_2018", "4시리즈_G22_2024_2025"),
    ("5008_2세대_2018_2019", "5008_2세대_2021_2024"),
    ("M5_F90_2018_2023", "5시리즈_G30_2017_2023"),
    ("더_뉴스포티지R_2014_2016", "5시리즈_G60_2024_2025"),
    ("6시리즈_GT_G32_2021_2024", "6시리즈_GT_G32_2018_2020"),
    ("그랜드카니발_2006_2010", "6시리즈_GT_G32_2018_2020"),
    ("박스터_718_2017_2024", "GLE_클래스_W167_2019_2024"),
    ("Q30_2017_2019", "911_992_2020_2024"),
    ("Q30_2017_2019", "G_클래스_W463b_2019_2025"),
    ("제네시스_DH_2014_2016", "G80_2017_2020"),
    ("카이엔_PO536_2019_2023", "G_클래스_W463b_2019_2025"),
    ("더_올뉴G80_2021_2024", "K5_2세대_2016_2018"),
    ("K5_3세대_2020_2023", "K5_하이브리드_3세대_2020_2023"),
    ("EQA_H243_2021_2024", "Q30_2017_2019"),
    ("뉴_ES300h_2013_2015", "Q50_2014_2017"),
    ("스포티지_4세대_2016_2018", "Q5_FY_2021_2024"),
    ("7시리즈_F01_2009_2015", "Q7_4M_2020_2023"),
    ("7시리즈_F01_2009_2015", "X4_F26_2015_2018"),
    ("E_클래스_W213_2017_2020", "S_클래스_W222_2014_2020"),
    ("C_클래스_W205_2015_2021", "S_클래스_W222_2014_2020"),
    ("G70_2018_2020", "S_클래스_W223_2021_2025"),
    ("더_뉴_G70_2021_2025", "X1_F48_2020_2022"),
    ("X3_G01_2022_2024", "X4_G02_2022_2025"),
    ("X6_G06_2020_2023", "X6_G06_2024_2025"),
    ("XC90_2세대_2020_2025", "XC90_2세대_2017_2019"),
    ("XM3_2024", "XM3_2020_2023"),
    ("RAV4_5세대_2019_2024", "그랜드_체로키_WL_2021_2023"),
    ("뉴_A6_2012_2014", "뉴_A6_2015_2018"),
    ("뉴_G80_2025_2026", "뉴_GV80_2024_2025"),
    ("그랜드카니발_2006_2010", "뉴_SM5_임프레션_2008_2010"),
    ("뉴_QM6_2021_2023", "더_뉴_QM6_2020_2023"),
    ("SM6_2016_2020", "더_뉴_SM6_2021_2024"),
    ("F150_2004_2021", "더_뉴_그랜드_스타렉스_2018_2021"),
    ("그랜드_스타렉스_2016_2018", "더_뉴_그랜드_스타렉스_2018_2021"),
    ("렉스턴_스포츠_칸_2019_2020", "더_뉴_렉스턴_스포츠_칸_2021_2025"),
    ("파나메라_2010_2016", "더_뉴_아반떼_2014_2016"),
    ("올_뉴_카니발_2015_2019", "더_뉴_카니발_2019_2020"),
    ("투싼_NX4_2021_2023", "더_뉴_투싼_NX4_2023_2025"),
    ("글래디에이터_JT_2020_2023", "랭글러_JL_2018_2024"),
    ("레니게이드_2015_2017", "레니게이드_2019_2023"),
    ("XJ_8세대_2010_2019", "머스탱_2015_2023"),
    ("그랜저_HG_2011_2014", "박스터_718_2017_2024"),
    ("디_올뉴싼타페_2024_2025", "싼타페_MX5_2024_2025"),
    ("SM7_뉴아트_2008_2011", "아베오_2012_2016"),
    ("올_뉴_K7_2016_2019", "올_뉴_K7_하이브리드_2017_2019"),
    ("카니발_4세대_2021", "카니발_4세대_2022_2023"),
    ("CLS_클래스_C257_2019_2023", "컨티넨탈_GT_3세대_2018_2023"),
    ("리얼_뉴_콜로라도_2021_2022", "콜로라도_2020_2020"),
    ("티볼리_아머_2018_2019", "티볼리_2015_2018"),
    ("파나메라_971_2017_2023", "파나메라_2010_2016"),
    ("All_New_XJ_2016_2019", "XJ_8세대_2010_2019"),
    ("그랜저_HG_2015_2017", "그랜저_HG_2011_2014"),
    ("아반떼_CN7_2021_2023", "더_뉴_아반떼_CN7_2023_2025")
]

# 2) 양방향 similarity_map 생성 (중복은 set 으로 걸러줍니다)
from collections import defaultdict

similarity_map = defaultdict(set)
for a, b in confusion_pairs:
    similarity_map[a].add(b)
    similarity_map[b].add(a)

# 3) 최종 dict 형태로 변환
similarity_map = {k: list(v) for k, v in similarity_map.items()}

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)  # 터미널에도 출력
        self.log.write(message)       # 파일에도 기록

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    def __init__(self, samples_list, transform=None, is_train=True):
        self.samples_list = samples_list
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        img_path, label = self.samples_list[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            dummy_tensor = torch.zeros((3, CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            return (dummy_tensor, 0) if self.is_train else (dummy_tensor, 0, img_path)
        
        # transform이 Albumentations인지 torchvision인지 구분
        if isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose):
            image = np.array(image)
            image = self.transform(image=image)['image']
        elif isinstance(self.transform, collections.abc.Callable):
            image = self.transform(image)

        return (image, label) if self.is_train else (image, label, img_path)


# --- 테스트 데이터셋 (inf.py에서도 사용) ---
class TestCustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
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
            return torch.zeros((3, CFG['IMG_SIZE'], CFG['IMG_SIZE']))
        if self.transform:
            image = self.transform(image)
        return image

# Model Define (inf.py에서도 사용)
class CustomTimmModel(nn.Module):
    def __init__(self, model_name, num_classes_to_predict, pretrained=True):
        super(CustomTimmModel, self).__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            self.feature_dim = self.backbone.num_features
        except Exception as e:
            print(f"Error creating model {model_name} with timm. Error: {e}")
            raise
        self.head = nn.Linear(self.feature_dim, num_classes_to_predict)
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# --- Albumentations 기반 이미지 변환 정의 ---
train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Affine(translate_percent=(0.1, 0.1), scale=(0.9, 1.1), shear=10, rotate=0, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def train_main():
    # work directory 생성
    work_dir = CFG['WORK_DIR']
    os.makedirs(work_dir, exist_ok=True)
    
    # logger
    sys.stdout = Logger(os.path.join(work_dir, "output.log"))
    
    # hyperparameter 저장
    with open(os.path.join(work_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=4, ensure_ascii=False)
    
    print("Using device:", device)
    print(f"Using model: {CFG['MODEL_NAME']}")
    if CFG['RUN_SINGLE_FOLD']:
        print(f"Running SINGLE FOLD mode: Target Fold = {CFG['TARGET_FOLD']}/{CFG['N_FOLDS']}")
    else:
        print(f"Running ALL {CFG['N_FOLDS']} FOLDS mode.")
    print(f"Early Stopping Patience: {CFG['EARLY_STOPPING_PATIENCE']}")

    seed_everything(CFG['SEED'])

    train_root = CFG['ROOT'] # 학습 데이터 경로
    initial_dataset = InitialCustomImageDataset(train_root)
    if not initial_dataset.samples:
        raise ValueError(f"No images found in {train_root}. Please check the path and data structure.")
    print(f"총 학습 이미지 수 (K-Fold 대상): {len(initial_dataset.samples)}")

    all_samples = initial_dataset.samples
    targets = [s[1] for s in all_samples]
    class_names = initial_dataset.classes
    num_classes = len(class_names)
    print(f"클래스: {class_names} (총 {num_classes}개)")

    # cutmix or mixup transform settings
    if CFG['CUTMIX'] and CFG["MIXUP"]:
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        print("매 배치마다 CUTMIX와 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
    elif CFG['CUTMIX']:
        cutmix_or_mixup = v2.CutMix(num_classes=num_classes)
        print("매 배치마다 CUTMIX를 랜덤하게 적용합니다. CFG를 확인하세요.")
    elif CFG['MIXUP']:
        cutmix_or_mixup = v2.MixUp(num_classes=num_classes)
        print("매 배치마다 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
    else:
        cutmix_or_mixup = None
    
    # 모델이 잘못 분류한 예시를 저장하기 위한 폴더 생성
    wrong_save_path = os.path.join(work_dir, "wrong_examples")
    os.makedirs(wrong_save_path, exist_ok=True)


    # class_names를 json 파일로 저장 (inf.py에서 사용)
    with open(os.path.join(work_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    print(f"Saved class_names to class_names.json")


    skf = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])
    overall_best_logloss = float('inf')
    overall_best_model_path = ""
    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_samples, targets)):
        fold_num = fold_idx + 1
        if CFG['RUN_SINGLE_FOLD'] and fold_num != CFG['TARGET_FOLD']:
            print(f"\nSkipping Fold {fold_num}/{CFG['N_FOLDS']} as RUN_SINGLE_FOLD is True and TARGET_FOLD is {CFG['TARGET_FOLD']}.")
            fold_results.append({'fold': fold_num, 'best_logloss': None, 'model_path': None, 'status': 'skipped'})
            continue
        print(f"\n===== Running Fold {fold_num}/{CFG['N_FOLDS']} =====\n")

        train_samples_fold = [all_samples[i] for i in train_indices]
        val_samples_fold = [all_samples[i] for i in val_indices]
        train_dataset_fold = FoldSpecificDataset(train_samples_fold, transform=train_transform)
        val_dataset_fold = FoldSpecificDataset(val_samples_fold, transform=val_transform, is_train=False)
        train_loader = DataLoader(train_dataset_fold, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
        print(f"Fold {fold_num}: Train images: {len(train_dataset_fold)}, Validation images: {len(val_dataset_fold)}")

        model = CustomTimmModel(model_name=CFG['MODEL_NAME'], num_classes_to_predict=num_classes).to(device)
        model_path = CFG['START_FROM']
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"{model_path} 모델을 불러와 해당 체크포인트부터 학습을 재개합니다. CFG를 확인해주세요.")
            print(f"Loaded model from {model_path}")
        else:
            print("체크포인트 경로가 없거나 제공되지 않았으므로 pretrained model으로부터 모델을 훈련시킵니다.")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=1e-2)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)

        best_logloss_fold = float('inf')
        current_fold_best_model_path = None
        patience_counter = 0
        best_val_loss_for_early_stopping = float('inf')

        for epoch in range(CFG['EPOCHS']):
            model.train()
            train_loss_epoch = 0.0
            # tqdm 생략 가능 (스크립트 실행 시) 또는 유지
            for images, labels in tqdm(train_loader, desc=f"[Fold {fold_num} Epoch {epoch+1}/{CFG['EPOCHS']}] Training", leave=False):
                images, labels = images.to(device), labels.to(device)

                # cutout을 위해 추가
                if CFG['CUTOUT']:
                    images = apply_cutout(images, mask_size = 64)
                
                # cutmix mixup을 위해 추가
                if cutmix_or_mixup:
                    images, labels = cutmix_or_mixup(images, labels)
                
    # --- 수정된 Mosaic 계열 증강 로직 ---
                applied_special_mosaic = False # 이번 배치에 Half 또는 Standard Mosaic이 적용되었는지 추적

                # 1. Mosaic 계열 증강을 적용할지 전체 확률(APPLY_MOSAIC_GROUP_P)로 결정
                if CFG.get('APPLY_MOSAIC_GROUP_P', 0.0) > 0 and \
                random.random() < CFG.get('APPLY_MOSAIC_GROUP_P'):
                    
                    can_apply_half_mosaic = CFG.get('HALF_MOSAIC_ENABLED', False)
                    can_apply_standard_mosaic = CFG.get('STANDARD_MOSAIC_ENABLED', False)

                    # 2. 어떤 Mosaic을 적용할지 결정
                    if can_apply_half_mosaic and can_apply_standard_mosaic:
                        # Half-Mosaic과 Standard Mosaic 모두 활성화된 경우, 50:50 확률로 선택
                        if random.random() < 0.5: 
                            orientation = random.choice(['horizontal', 'vertical'])
                            images, labels = half_mosaic( # 사용자 정의 half_mosaic 호출
                                images, labels,
                                class_names, similarity_map, 
                                num_classes,
                                orientation=orientation
                            )
                            applied_special_mosaic = True
                        else: 
                            images, labels = mosaic_augmentation( # 사용자 정의 mosaic_augmentation 호출
                                images, labels, num_classes # train_main에서 정의된 num_classes 사용
                            )
                            applied_special_mosaic = True
                    elif can_apply_half_mosaic: # Half-Mosaic만 활성화된 경우
                        orientation = random.choice(['horizontal', 'vertical'])
                        images, labels = half_mosaic(
                            images, labels,
                            class_names, similarity_map,
                            num_classes,
                            orientation=orientation
                        )
                        applied_special_mosaic = True
                    elif can_apply_standard_mosaic: # Standard Mosaic만 활성화된 경우
                        images, labels = mosaic_augmentation(
                            images, labels, num_classes
                        )
                        applied_special_mosaic = True

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            avg_train_loss_epoch = train_loss_epoch / len(train_loader)

            model.eval()
            val_loss_epoch = 0.0
            correct_epoch = 0
            total_epoch = 0
            all_probs_epoch = []
            all_labels_epoch = []
            wrong_img_dict = defaultdict(list)
            with torch.no_grad():
                for images, labels, img_paths in tqdm(val_loader, desc=f"[Fold {fold_num} Epoch {epoch+1}/{CFG['EPOCHS']}] Validation", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss_epoch += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_epoch += (preds == labels).sum().item()
                    total_epoch += labels.size(0)
                    probs = F.softmax(outputs, dim=1)
                    all_probs_epoch.extend(probs.cpu().numpy())
                    all_labels_epoch.extend(labels.cpu().numpy())
                    
                    # === 틀린 예측 탐색 ===
                    wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]
                    for i in wrong_indices: # 인덱스 변수명 변경 (선택 사항)
                        true_label_idx = labels[i].item()
                        predicted_label_idx = preds[i].item()

                        image_path_for_wrong = img_paths[i]
                        true_class_name_for_wrong = class_names[true_label_idx]
                        predicted_class_name_for_wrong = class_names[predicted_label_idx]

                        # JSON 키를 모델이 예측한 클래스명으로 변경
                        wrong_img_dict[predicted_class_name_for_wrong].append({
                            'image_path': image_path_for_wrong,
                            'correct_answer': true_class_name_for_wrong # 필드명도 이전과 동일하게
                        })
                with open(os.path.join(wrong_save_path, f"Fold_{fold_num}_Epoch_{epoch+1}_wrong_examples.json"), "w", encoding="utf-8") as f:
                    json.dump(wrong_img_dict, f, indent=4, ensure_ascii=False)
                    
            avg_val_loss_epoch = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else float('inf')
            val_accuracy_epoch = 100 * correct_epoch / total_epoch if total_epoch > 0 else 0
            val_logloss_epoch = log_loss(all_labels_epoch, all_probs_epoch, labels=list(range(num_classes))) if total_epoch > 0 and len(np.unique(all_labels_epoch)) > 1 else float('inf')

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Fold {fold_num} Epoch {epoch+1} - Train Loss: {avg_train_loss_epoch:.4f} | Valid Loss: {avg_val_loss_epoch:.4f} | Valid Acc: {val_accuracy_epoch:.2f}% | Valid LogLoss: {val_logloss_epoch:.4f} | LR: {current_lr:.1e}")
            scheduler.step()

            if val_logloss_epoch < best_logloss_fold:
                best_logloss_fold = val_logloss_epoch
                current_fold_best_model_path = os.path.join(work_dir, f'best_model_{CFG["MODEL_NAME"]}_fold{fold_num}.pth')
                torch.save(model.state_dict(), current_fold_best_model_path)
                print(f"Fold {fold_num} 📦 Best model saved at epoch {epoch+1} (LogLoss: {best_logloss_fold:.4f}) to {current_fold_best_model_path}")

            if val_logloss_epoch < best_val_loss_for_early_stopping:
                best_val_loss_for_early_stopping = val_logloss_epoch
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= CFG['EARLY_STOPPING_PATIENCE']:
                print(f"Fold {fold_num} Early stopping triggered at epoch {epoch+1}.")
                break
        
        fold_results.append({'fold': fold_num, 'best_logloss': best_logloss_fold if best_logloss_fold != float('inf') else None, 'model_path': current_fold_best_model_path, 'status': 'completed'})
        if current_fold_best_model_path and best_logloss_fold < overall_best_logloss:
            overall_best_logloss = best_logloss_fold
            overall_best_model_path = current_fold_best_model_path
            print(f"🌟 New Overall Best Model from Fold {fold_num} (LogLoss: {overall_best_logloss:.4f}, Path: {overall_best_model_path})")

    print("\n===== K-Fold Cross Validation Summary =====")
    # ... (결과 요약 부분은 동일하게 유지)
    total_logloss_sum = 0
    valid_folds_count = 0
    executed_folds_count = 0
    for res in fold_results:
        if res['status'] == 'completed':
            executed_folds_count +=1
            print(f"Fold {res['fold']}: Best LogLoss = {res['best_logloss']:.4f if res['best_logloss'] else 'N/A'}, Model Path = {res['model_path'] if res['model_path'] else 'N/A'} ({res['status']})")
            if res['best_logloss']:
                total_logloss_sum += res['best_logloss']
                valid_folds_count +=1
        elif res['status'] == 'skipped':
            print(f"Fold {res['fold']}: Status = {res['status']}")

    if executed_folds_count > 0 and valid_folds_count > 0:
        avg_logloss_executed_folds = total_logloss_sum / valid_folds_count
        print(f"\nAverage Best LogLoss across {valid_folds_count} successfully completed and valid folds: {avg_logloss_executed_folds:.4f}")
    elif executed_folds_count > 0:
        print("\nNo folds successfully saved a model to calculate average logloss.")
    else:
        print("\nNo folds were executed.")

    print(f"\nOverall Best LogLoss (among executed folds): {overall_best_logloss:.4f if overall_best_logloss != float('inf') else 'N/A'}")
    print(f"Path to the overall best model for inference: {overall_best_model_path if overall_best_model_path else 'N/A'}")
    print("Training finished.")


if __name__ == '__main__':
    train_main()