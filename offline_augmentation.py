import os
import random
import torch
import pandas as pd
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataset import *
from augmentations import *
from sklearn.model_selection import StratifiedKFold

CFG = {
    "ROOT": '/home/sh/hecho/train',
    "WORK_DIR": '/home/sh/hecho/KD_test_folder/offline_aug_test',
    # 해당 augmentation들은 선택된 것들 중 랜덤하게 '1개'만 적용이 됩니다(배치마다 랜덤하게 1개 선택)
    "CUTMIX": True,
    "MIXUP":  True,
    "MOSAIC": True,
    "CUTOUT": False,
    #################
    
    'SEED': 42,
    'N_FOLDS': 5,
    'TARGET_FOLD': 1,
    "BATCH_SIZE": 64,
    'augmentation_time_per_batch': 2,
    'IMG_SIZE': 700
}

# 이미지 변환 정의 (val_transform은 inf.py에서도 유사하게 사용)
train_transform = transforms.Compose([
    transforms.Resize(size=(CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    v2.AugMix(severity=4),
    transforms.ToTensor(),
])

@torch.no_grad()
def save_augmented_dataset_with_original_names(dataloader, save_root, label_map, augmentation_time_per_batch, device=torch.device('cpu')):
    """
    Args:
        dataloader: yielding (images, labels, img_paths)
        save_root: 저장 루트 경로 (e.g., "./augmented_train")
        label_map: {class_index: class_name}
        device: torch.device
    """
    num_classes = len(label_map)
    os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
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
    # 복잡한 augmentation의 경우 여러개 선택 시 하나만 적용하기 위한 list
    target_augmentations = ["CUTMIX", "MIXUP", "MOSAIC", "CUTOUT"]
    selected_augmentations = [i for i in target_augmentations if CFG[i]]

    all_ids = []
    all_labels = []

    for images, labels, img_paths in tqdm(dataloader, desc='Saving images'):
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=num_classes).float().to(labels.device)
        
        for i in range(augmentation_time_per_batch):
            tmp_images = images.detach().clone().to(device)
            tmp_labels = labels.detach().clone().to(device)
            
            if selected_augmentations:
                choice = random.choice(selected_augmentations)
            else:
                choice = None
                
            # cutout을 위해 추가
            if CFG['CUTOUT'] and choice == 'CUTOUT':
                tmp_images = apply_cutout(tmp_images, mask_size = 64)
            
            # cutmix mixup을 위해 추가
            if cutmix_or_mixup and (choice == 'MIXUP' or choice == 'CUTMIX'):
                tmp_images, tmp_labels = cutmix_or_mixup(tmp_images, tmp_labels)
            
            # MOSAIC을 위해 추가
            if CFG['MOSAIC'] and (choice == 'MOSAIC'):
                tmp_images, tmp_labels = apply_mosaic(tmp_images, tmp_labels, num_classes)

            for j in range(tmp_images.size(0)):
                img_tensor = tmp_images[j].cpu()
                class_vector = tmp_labels[j].cpu()

                # 원본 이미지 파일명 추출
                original_filename = f"{i}_" + os.path.basename(img_paths[j])
                file_path = os.path.join(save_root, 'images', original_filename)

                # 저장
                save_image(img_tensor, file_path)

                all_ids.append(original_filename)
                all_labels.append(class_vector)

    # CSV 저장
    label_array = np.stack(all_labels)  # shape [N, num_classes]
    df = pd.DataFrame(label_array, columns=[f"class_{i}" for i in range(num_classes)])
    df.insert(0, "ID", all_ids)  # 첫 번째 열에 ID 삽입
    df.to_csv(os.path.join(save_root, "labels.csv"), index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} images to {save_root}/images and metadata to labels.csv")


def main():
    work_dir = CFG['WORK_DIR']
    
    train_root = CFG['ROOT'] # 학습 데이터 경로
    initial_dataset = InitialCustomImageDataset(train_root)
    if not initial_dataset.samples:
        raise ValueError(f"No images found in {train_root}. Please check the path and data structure.")
    print(f"총 학습 이미지 수 (K-Fold 대상): {len(initial_dataset.samples)}")

    all_samples = initial_dataset.samples
    targets = [s[1] for s in all_samples]
    class_names = initial_dataset.classes
    label_map = {i: name for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"클래스: {class_names} (총 {num_classes}개)")
    
    skf = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])
    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_samples, targets)):
        fold_num = fold_idx + 1
        if fold_num != CFG['TARGET_FOLD']:
            print(f"\nSkipping Fold {fold_num}/{CFG['N_FOLDS']} as RUN_SINGLE_FOLD is True and TARGET_FOLD is {CFG['TARGET_FOLD']}.")
            fold_results.append({'fold': fold_num, 'best_logloss': None, 'model_path': None, 'status': 'skipped'})
            continue
        print(f"\n===== Running Fold {fold_num}/{CFG['N_FOLDS']} =====\n")

        train_samples_fold = [all_samples[i] for i in train_indices]
        train_dataset_fold = FoldSpecificDataset(train_samples_fold, image_size = CFG['IMG_SIZE'], transform=train_transform, is_train=False)
        train_loader = DataLoader(train_dataset_fold, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
        save_augmented_dataset_with_original_names(train_loader, work_dir, label_map, CFG['augmentation_time_per_batch'])

if __name__=="__main__":
    main()