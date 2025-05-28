import os
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json # class_names 로드를 위해 추가
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# train.py에서 필요한 클래스와 함수, CFG를 가져옴
from albu_train import CustomTimmModel
from dataset import InitialCustomImageDataset, FoldSpecificDataset
from utils import *

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter Setting
CFG = {
    "WORK_DIR": '/project/ahnailab/jys0207/CP/tjrgus5/hecto/work_directories/convnext_cutmix+mixup_test', # train.py로 생성된 work_directory
    "MODEL_PATH": None, # 사용할 모델 경로. None이면 WORK_DIR에서 임의로 .pth파일을 찾아서 고름
    "ROOT": '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/train', # data_path
    "BATCH_SIZE": 32
}

def validate_for_all_train_data():
    print("Using device:", device)
    
    # work_directory
    work_dir = CFG['WORK_DIR']
    
    # 몇몇 세팅 통일을 위해 train CFG 가져오기
    with open(os.path.join(work_dir, "settings.json"), "r") as f:
        TRAIN_CFG = json.load(f)
    # 제외한 데이터만 업데이트
    filtered_data = {k: v for k, v in TRAIN_CFG.items() if k not in CFG.keys()}
    CFG.update(filtered_data)
    print(f"Inference CFG: {CFG}")
    seed_everything(CFG['SEED'])
    
    # transform 정의
    val_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # class_names 로드
    try:
        with open(os.path.join(work_dir, 'class_names.json'), 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Error: class_names.json not found. Please run train.py first to generate it.")
        return
    num_classes = len(class_names)
    print(f"Loaded class_names: {class_names} (Total: {num_classes})")


    # 테스트 데이터셋 및 DataLoader
    val_root = CFG['ROOT'] # 테스트 데이터 경로
    initial_dataset = InitialCustomImageDataset(val_root)
    if not initial_dataset.samples:
        raise ValueError(f"No images found in {val_root}. Please check the path and data structure.")
    print(f"총 학습 이미지 수 (K-Fold 대상): {len(initial_dataset.samples)}")

    all_samples = initial_dataset.samples
    print(f"클래스: {class_names} (총 {num_classes}개)")
    val_dataset = FoldSpecificDataset(all_samples, image_size=CFG['IMG_SIZE'], transform=val_transform, is_train=False) # train.py의 val_transform과 동일한 것을 사용
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)

    # 모델 로드
    model = CustomTimmModel(model_name=CFG['MODEL_NAME'], num_classes_to_predict=num_classes).to(device)
    
    model_path = CFG['MODEL_PATH'] if CFG['MODEL_PATH'] else find_first_file_by_extension(work_dir)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please check the path in CFG_INF['MODEL_PATH'].")
        print("You might need to run train.py first or update the path to the desired .pth file.")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the model architecture in train.py (CustomTimmModel) matches the saved model.")
        return
    
    # val_loss 계산을 위해 선언
    criterion = nn.CrossEntropyLoss()

    model.eval()
    num_corrects = 0
    wrong_img_dict = defaultdict(list)
    with torch.no_grad():
        for images, labels, img_paths in tqdm(val_loader, desc="Inference"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            num_corrects += (preds == labels).sum().item()
            # === 틀린 예측 탐색 ===
            wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]  # 틀린 인덱스만 추출
            for idx in wrong_indices:
                path = img_paths[idx]  # 예: 'data/train/cat/image1.jpg'
                parent_folder = os.path.basename(os.path.dirname(path))  # 예: 'cat'
                wrong_img_dict[parent_folder].append({
                    'image_path': path,
                    'model_answer': class_names[preds[idx]]
                })
    with open(os.path.join(work_dir, "all_wrong_examples.json"), "w", encoding="utf-8") as f:
        json.dump(wrong_img_dict, f, indent=4, ensure_ascii=False)
    

if __name__ == '__main__':
    validate_for_all_train_data()