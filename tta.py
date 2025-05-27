# 현재 모듈화 하고 나서 수정 안해서 아마 제대로 작동 안할듯

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json # class_names 로드를 위해 추가
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
# train.py에서 필요한 클래스와 함수, CFG를 가져옴
from albu_train import CustomTimmModel, CFG as TRAIN_CFG
from utils import *

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference Configuration
CFG_INF = {
    'MODEL_PATH': '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/checkpoint/0519/best_model_convnext_base(1214).pth', # 학습 후 생성된 실제 모델 경로로 수정 필요
    'ROOT': '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/test',
    'SUBMISSION_FILE': '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/sample_submission.csv',
    'BATCH_SIZE': 64, # 추론 시 배치 크기
    'TTA_TIMES': 4, # tta 수행 횟수
    # MODEL_NAME, IMG_SIZE 등은 TRAIN_CFG에서 가져와 사용
    'MODEL_NAME': TRAIN_CFG['MODEL_NAME'],
    'IMG_SIZE': TRAIN_CFG['IMG_SIZE'],
    'SEED': TRAIN_CFG['SEED'] # 일관성을 위해 시드 설정
}

# 이미지 변환 정의 (val_transform은 inf.py에서도 유사하게 사용)
tta_transform = transforms.Compose([
    transforms.Resize((CFG_INF['IMG_SIZE'], CFG_INF['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TTATestCustomImageDataset(Dataset):
    def __init__(self, root_dir, transform, tta_times=4):
        """
        Args:
            root_dir (str): 테스트 이미지가 있는 폴더 경로
            transform (Transform): 단일 transform (e.g. train_transform)
            tta_times (int): 동일 이미지에 몇 번 transform을 적용할지
        """
        self.root_dir = root_dir
        self.transform = transform
        self.tta_times = tta_times
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
            return [torch.zeros((3, CFG_INF['IMG_SIZE'], CFG_INF['IMG_SIZE'])) for _ in range(self.tta_times)]

        images = [self.transform(image) for _ in range(self.tta_times)]
        return images  # (tta_times, C, H, W)


def inference_main():
    print("Using device:", device)
    print(f"Inference CFG: {CFG_INF}")
    seed_everything(CFG_INF['SEED'])

    # class_names 로드
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Error: class_names.json not found. Please run train.py first to generate it.")
        return
    num_classes = len(class_names)
    print(f"Loaded class_names: {class_names} (Total: {num_classes})")


    # 테스트 데이터셋 및 DataLoader
    test_root = CFG_INF['ROOT'] # 테스트 데이터 경로
    test_dataset = TTATestCustomImageDataset(test_root, transform=tta_transform, tta_times=CFG_INF['TTA_TIMES']) # train.py의 val_transform과 동일한 것을 사용
    
    if not test_dataset.samples:
        print(f"No images found in {test_root} for inference. Skipping submission file generation.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=CFG_INF['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)

    # 모델 로드
    model = CustomTimmModel(model_name=CFG_INF['MODEL_NAME'], num_classes_to_predict=num_classes).to(device)
    
    model_path = CFG_INF['MODEL_PATH']
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

    model.eval()
    results = []
    with torch.no_grad():
        for tta_images in tqdm(test_loader, desc="Inference"):
            # tta_images: list of batch, each item is list of T tensors
            # Shape: [T, B, C, H, W]  if collate_fn를 안 썼다면, 기본은 list[Tensor]

            # 배치 내 각 샘플별 TTA 이미지 리스트로 구성된 경우 처리
            tta_probs = 0
            for sample_images in tta_images:  # sample_images: List[Tensor], len == tta_times
                images = sample_images.to(device)  # (1, C, H, W)
                output = model(images)
                tta_probs += (F.softmax(output, dim=1) / CFG_INF['TTA_TIMES'])

            # 하나의 배치로 합치기
            results.extend(tta_probs.detach().cpu().numpy())

    pred_df = pd.DataFrame(results, columns=class_names)
    pred_df.to_csv("./tmp.csv")

    # Submission 파일 생성
    submission_file_path = CFG_INF['SUBMISSION_FILE'] # 샘플 제출 파일 경로
    try:
        submission_df = pd.read_csv(submission_file_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Sample submission file {submission_file_path} not found.")
        return

    class_columns_in_submission = submission_df.columns[1:]
    try:
        submission_df[class_columns_in_submission] = pred_df[class_columns_in_submission].values
    except KeyError as e:
        print(f"KeyError during submission assignment: {e}")
        print(f"Make sure class_names.json and sample_submission.csv columns align with model output.")
        return
    except ValueError as e:
        print(f"ValueError during submission assignment: {e}")
        return


    # 출력 파일명에 모델 이름과 (단일 폴드 실행 시) 폴드 번호 또는 'best_overall' 포함
    base_output_filename = f'submission_{CFG_INF["MODEL_NAME"]}'
    # TRAIN_CFG에서 RUN_SINGLE_FOLD와 TARGET_FOLD를 참조하여 파일명 결정
    if TRAIN_CFG.get('RUN_SINGLE_FOLD', False) and TRAIN_CFG.get('TARGET_FOLD') is not None and CFG_INF['MODEL_PATH'].endswith(f"_fold{TRAIN_CFG['TARGET_FOLD']}.pth"):
         base_output_filename += f'_fold{TRAIN_CFG["TARGET_FOLD"]}'
    elif "_fold" in CFG_INF['MODEL_PATH']: # 경로에 fold 정보가 있다면 사용
        try:
            fold_num_from_path = CFG_INF['MODEL_PATH'].split('_fold')[1].split('.pth')[0]
            base_output_filename += f'_fold{fold_num_from_path}'
        except: # 경로 파싱 실패 시 일반 이름 사용
            if "overall" not in CFG_INF['MODEL_PATH'].lower(): # overall이 아니면 경로명 일부 사용 시도
                 path_part = os.path.basename(CFG_INF['MODEL_PATH']).replace('.pth','').replace('best_model_','')
                 base_output_filename = f'submission_{path_part}'


    output_submission_filename = f'{base_output_filename}.csv'

    submission_df.to_csv(output_submission_filename, index=False, encoding='utf-8-sig')
    print(f"🎉 Submission file created successfully: {output_submission_filename}")

if __name__ == '__main__':
    inference_main()