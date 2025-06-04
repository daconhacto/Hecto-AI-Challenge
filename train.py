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
from augmentations import *
from utils import *
from dataset import *
from model import *

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter Setting
CFG = {
    "ROOT": '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/train',
    "WORK_DIR": '/project/ahnailab/jys0207/CP/tjrgus5/hecto/work_directories/logging_test',

    # retraining 설정
    "START_FROM": None, # 만약 None이 아닌 .pth파일 경로 입력하면 해당 checkpoint를 load해서 시작
    "GROUP_PATH": None, # 만약 None이 아닌 group.json의 경로르 입력하면 해당 class들만 활용하여 train을 진행함
    
    # wrong example을 뽑을 threshold 조건. threshold 이하인 confidence를 가지는 케이스를 저장.
    "WRONG_THRESHOLD": 0.7,
    "GROUP_JSON_START_EPOCH": 5, # work_dir에 해당 에폭부터의 wrong_examples를 통합한 json파일을 저장하게됩니다.

    # 해당 augmentation들은 선택된 것들 중 랜덤하게 '1개'만 적용이 됩니다(배치마다 랜덤하게 1개 선택)
    "CUTMIX": {
        'enable': True,
        'params':{'alpha':1.0} # alpha값 float로 정의 안하면 오류남
    },
    "MIXUP": {
        'enable': True,
        'params':{'alpha':1.0} # alpha값 float로 정의 안하면 오류남
    },
    "MOSAIC": {
        'enable': True,
        'params':{
            'p': 1.0,
            'grid_size': 2,
            'use_saliency': True
        }
    },
    "CUTOUT": {
        'enable': False,
        'params':{
            'mask_size': 32
        }
    },

    # 기타 설정값들
    'IMG_SIZE': 448, # Number or Tuple(Height, Width)
    'BATCH_SIZE': 32, # 학습 시 배치 크기
    'EPOCHS': 25,
    'SEED' : 42,
    'MODEL_NAME': 'convnext_base.fb_in22k_ft_in1k_384', # 사용할 모델 이름
    'N_FOLDS': 5,
    'EARLY_STOPPING_PATIENCE': 3,
    'RUN_SINGLE_FOLD': True,  # True로 설정 시 특정 폴드만 실행
    'TARGET_FOLD': 1,          # RUN_SINGLE_FOLD가 True일 때 실행할 폴드 번호 (1-based)
    

    # 새롭게 추가된 logging파트. class의 경우 무조건 풀경로로 적어야합니다. nn.CrossEntropyLoss 처럼 적으면 오류남
    'LOSS': {
        'class': 'torch.nn.CrossEntropyLoss',
        'params': {}   
    },
    'OPTIMIZER': {
        'class': 'torch.optim.AdamW',
        'params': {
            'lr': 1e-4,
            'weight_decay': 1e-2
        }
    },
    'SCHEDULER': {
        'class': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        'params': {
            'mode': 'min',
            'factor': 0.1,
            'patience':2
        }
    },
}

CFG['IMG_SIZE'] = CFG['IMG_SIZE'] if isinstance(CFG['IMG_SIZE'], tuple) else (CFG['IMG_SIZE'], CFG['IMG_SIZE'])
# 이미지 변환 정의 Albumentation하고 torchvision 어느 라이브러리를 활용하든 상관없이 적용 가능
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'][0], CFG['IMG_SIZE'][1])),
    v2.AugMix(severity=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([ # inf.py의 test_transform과 동일해야 함
    transforms.Resize((CFG['IMG_SIZE'][0], CFG['IMG_SIZE'][1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    # transform setting 저장
    save_transform(train_transform, os.path.join(work_dir, "train_transform.json"))
    save_transform(val_transform, os.path.join(work_dir, "val_transform.json"))
    
    
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
    if CFG['CUTMIX']['enable'] and CFG["MIXUP"]['enable']:
        cutmix = v2.CutMix(num_classes=num_classes, **CFG['CUTMIX']['params'])
        mixup = v2.MixUp(num_classes=num_classes, **CFG['MIXUP']['params'])
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        print("매 배치마다 CUTMIX와 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
    elif CFG['CUTMIX']['enable']:
        cutmix_or_mixup = v2.CutMix(num_classes=num_classes, **CFG['CUTMIX']['params'])
        print("매 배치마다 CUTMIX를 랜덤하게 적용합니다. CFG를 확인하세요.")
    elif CFG["MIXUP"]['enable']:
        cutmix_or_mixup = v2.MixUp(num_classes=num_classes, **CFG['MIXUP']['params'])
        print("매 배치마다 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
    else:
        cutmix_or_mixup = None

    # 복잡한 augmentation의 경우 여러개 선택 시 하나만 적용하기 위한 list
    target_augmentations = ["CUTMIX", "MIXUP", "MOSAIC", "CUTOUT"]
    selected_augmentations = [i for i in target_augmentations if CFG[i]]
    
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
        train_dataset_fold = FoldSpecificDataset(train_samples_fold, image_size = CFG['IMG_SIZE'], transform=train_transform)
        val_dataset_fold = FoldSpecificDataset(val_samples_fold, image_size = CFG['IMG_SIZE'], transform=val_transform, is_train=False)

        # group_path가 설정되어 있으면 해당 class들로만 훈련을 진행
        # group 로드
        if CFG['GROUP_PATH']:
            with open(CFG['GROUP_PATH'], 'r') as f:
                wrong_example_group = json.load(f)
            wrong_example_group = convert_classname_groups_to_index_groups(wrong_example_group, class_names)
            # difficult example sampling을 위한 전처리 과정
            label_to_indices = build_class_index_map(train_samples_fold)
            sampler = GroupedBatchSampler(label_to_indices, wrong_example_group, CFG['BATCH_SIZE'])
            train_loader = DataLoader(train_dataset_fold, num_workers=2, pin_memory=True, batch_sampler=sampler)
        else:
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
        
        criterion = get_class_from_string(CFG['LOSS']['class'])(**CFG['LOSS']['params'])
        optimizer = get_class_from_string(CFG['OPTIMIZER']['class'])(model.parameters(), **CFG['OPTIMIZER']['params'])
        scheduler = get_class_from_string(CFG['SCHEDULER']['class'])(optimizer, **CFG['SCHEDULER']['params'])

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

                if selected_augmentations:
                    choice = random.choice(selected_augmentations)
                else:
                    choice = None
                    
                # cutout을 위해 추가
                if CFG['CUTOUT']['enable'] and choice == 'CUTOUT':
                    images = apply_cutout(images, **CFG['CUTOUT']['params'])
                
                # cutmix mixup을 위해 추가
                if cutmix_or_mixup and (choice == 'MIXUP' or choice == 'CUTMIX'):
                    images, labels = cutmix_or_mixup(images, labels)
                
                # MOSAIC을 위해 추가
                if CFG['MOSAIC']['enable'] and (choice == 'MOSAIC'):
                    images, labels = apply_mosaic(images, labels, num_classes, **CFG['MOSAIC']['params'])

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
                    correct_class_confidences = probs[torch.arange(len(labels)), labels]
                    wrong_indices = (correct_class_confidences <= CFG['WRONG_THRESHOLD']).nonzero(as_tuple=True)[0]  # 틀린 인덱스만 추출
                    for idx in wrong_indices:
                        path = img_paths[idx]  # 예: 'data/train/cat/image1.jpg'
                        parent_folder = os.path.basename(os.path.dirname(path))  # 예: 'cat'
                        pred = preds[idx]
                        label = labels[idx]

                        if pred == label:
                            # 예측은 맞았지만 confidence가 낮음 → 두 번째로 높은 클래스 선택
                            sorted_probs, sorted_indices = probs[idx].sort(descending=True)
                            second_best_class = sorted_indices[1].item()
                            model_answer = class_names[second_best_class]
                        else:
                            # 아예 틀린 예측 → 기존대로 예측 결과 사용
                            model_answer = class_names[pred]

                        wrong_img_dict[parent_folder].append({
                            'image_path': path,
                            'model_answer': model_answer
                        })
                with open(os.path.join(wrong_save_path, f"Fold_{fold_num}_Epoch_{epoch+1}_wrong_examples.json"), "w", encoding="utf-8") as f:
                    json.dump(wrong_img_dict, f, indent=4, ensure_ascii=False)
                    
            avg_val_loss_epoch = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else float('inf')
            val_accuracy_epoch = 100 * correct_epoch / total_epoch if total_epoch > 0 else 0
            val_logloss_epoch = log_loss(all_labels_epoch, all_probs_epoch, labels=list(range(num_classes))) if total_epoch > 0 and len(np.unique(all_labels_epoch)) > 1 else float('inf')

            if CFG['SCHEDULER']['class'] == 'torch.optim.lr_scheduler.ReduceLROnPlateau':
                scheduler.step(val_logloss_epoch)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Fold {fold_num} Epoch {epoch+1} - Train Loss: {avg_train_loss_epoch:.4f} | Valid Loss: {avg_val_loss_epoch:.4f} | Valid Acc: {val_accuracy_epoch:.2f}% | Valid LogLoss: {val_logloss_epoch:.4f} | LR: {current_lr:.1e}")

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
        # 전체 틀린 그룹을 저장
        get_total_wrong_groups(work_dir, CFG['GROUP_JSON_START_EPOCH'], fold_num)
        
    print("\n===== K-Fold Cross Validation Summary =====")
    # ... (결과 요약 부분은 동일하게 유지)
    total_logloss_sum = 0
    valid_folds_count = 0
    executed_folds_count = 0
    for res in fold_results:
        if res['status'] == 'completed':
            executed_folds_count +=1
            print(f"Fold {res['fold']}: Best LogLoss = {res['best_logloss'] if res['best_logloss'] else 'N/A'}, Model Path = {res['model_path'] if res['model_path'] else 'N/A'} ({res['status']})")
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

    print(f"\nOverall Best LogLoss (among executed folds): {overall_best_logloss if overall_best_logloss != float('inf') else 'N/A'}")
    print(f"Path to the overall best model for inference: {overall_best_model_path if overall_best_model_path else 'N/A'}")
    print("Training finished.")


if __name__ == '__main__':
    train_main()