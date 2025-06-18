import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import timm # pip install timm
import random

CFG = {
    # --- 공통 설정 ---
    'BATCH_SIZE': 16,
    'SEED': 42,
    'CLASS_NAMES_PATH': './class_names.json',  # train.py 실행 후 생성된 class_names.json 경로
    'TEST_ROOT': './data/test',                    # 테스트 이미지 폴더 경로
    'SAMPLE_SUBMISSION_PATH': './data/sample_submission.csv', # 샘플 제출 파일 경로

    # --- 1단계 추론 및 앙상블 설정 ---
    'STAGE1_GROUPS': [
        {
            "group_name": "ConvNext Ensemble",
            "output_csv_path": "stage1_convnext_ensemble.csv",
            # 이 그룹 전체의 대표 LogLoss 값 (2단계 앙상블 가중치 계산에 사용)
            "ensemble_log_loss": 0.09121,
            "models": [
                # ConvNext 5개 Fold에 대한 정보를 입력하세요.
                {"path": "./pth_files/conv_fold1.pth", "name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384", "img_size": 600, "val_log_loss": 0.09476},
                {"path": "./pth_files/conv_fold2.pth", "name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384", "img_size": 600, "val_log_loss": 0.10195},
                {"path": "./pth_files/conv_fold3.pth", "name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384", "img_size": 600, "val_log_loss": 0.10860},
                {"path": "./pth_files/conv_fold4.pth", "name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384", "img_size": 600, "val_log_loss": 0.10146},
                {"path": "./pth_files/conv_fold5.pth", "name": "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384", "img_size": 600, "val_log_loss": 0.10181},
            ]
        },
        {
            "group_name": "EVA Ensemble",
            "output_csv_path": "stage1_eva_ensemble.csv",
            # 이 그룹 전체의 대표 LogLoss 값 (2단계 앙상블 가중치 계산에 사용)
            "ensemble_log_loss": 0.10367,
            "models": [
                # EVA 3개 Fold에 대한 정보를 입력하세요. (가중치: 0.13874, 0.112754, 0.1089)
                {"path": "./pth_files/eva_fold1.pth", "name": "eva02_large_patch14_448.mim_m38m_ft_in1k", "img_size": 448, "val_log_loss": 0.13874},
                {"path": "./pth_files/eva_fold2.pth", "name": "eva02_large_patch14_448.mim_m38m_ft_in1k", "img_size": 448, "val_log_loss": 0.12754},
                {"path": "./pth_files/eva_fold3.pth", "name": "eva02_large_patch14_448.mim_m38m_ft_in1k", "img_size": 448, "val_log_loss": 0.11089},
            ]
        }
    ],

    # --- 2단계 CSV 앙상블 설정 ---
    'STAGE2_OUTPUT_NAME': 'final_submission.csv',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomTimmModel(torch.nn.Module):
    def __init__(self, model_name, num_classes_to_predict, pretrained=True):
        super(CustomTimmModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.feature_dim = self.backbone.num_features
        self.head = torch.nn.Linear(self.feature_dim, num_classes_to_predict)
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class TestCustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            print(f"🚨 경고: 테스트 디렉토리 {root_dir}를 찾을 수 없습니다.")
            return
        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root_dir, fname)
                self.samples.append((img_path,))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_test_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def compute_weights(items_with_loss, loss_key='val_log_loss'):
    """val_log_loss 리스트를 기반으로 가중치를 계산합니다."""
    losses = [info.get(loss_key) for info in items_with_loss]
    if any(l is None or l <= 0 for l in losses):
        print("⚠️ val_log_loss가 없거나 0 이하인 항목이 있어 동일 가중치를 적용합니다.")
        return np.ones(len(items_with_loss)) / len(items_with_loss)
    
    inv_losses = 1.0 / np.array(losses)
    weights = inv_losses / inv_losses.sum()
    return weights

def run_stage1_inference_ensemble(group_config, num_classes):
    """지정된 모델 그룹에 대한 추론 및 가중 평균 앙상블을 수행하고 결과를 CSV로 저장합니다."""
    print("\n" + "="*80)
    print(f"🚀 STAGE 1: '{group_config['group_name']}' 그룹 처리 시작")
    print("="*80)

    all_model_predictions = []
    
    current_img_size_for_loader = -1
    test_loader = None

    for model_info in group_config['models']:
        model_path = model_info['path']
        model_arch = model_info['name']
        model_img_size = model_info['img_size']

        if not os.path.exists(model_path):
            print(f"🚨 경고: 모델 경로 {model_path}를 찾을 수 없습니다. 이 모델은 건너뜁니다.")
            continue

        # 이미지 크기에 맞는 DataLoader 생성 또는 재사용
        if test_loader is None or current_img_size_for_loader != model_img_size:
            print(f"\n🔄 이미지 크기 {model_img_size}에 대한 DataLoader를 생성합니다...")
            transform = get_test_transform(model_img_size)
            dataset = TestCustomImageDataset(CFG['TEST_ROOT'], transform=transform)
            if not dataset.samples:
                raise FileNotFoundError(f"테스트 폴더 '{CFG['TEST_ROOT']}'에 이미지가 없습니다.")
            test_loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
            current_img_size_for_loader = model_img_size
            print(f"✅ DataLoader 준비 완료. ({len(dataset)}개 이미지)")

        # 단일 모델 추론
        print(f"  - 추론 모델: {os.path.basename(model_path)} (Arch: {model_arch}, ImgSize: {model_img_size})")
        model = CustomTimmModel(model_name=model_arch, num_classes_to_predict=num_classes).to(device)
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.to(device)
        except Exception as e:
            print(f"🚨 모델 로딩 실패: {e}. 이 모델은 건너뜁니다.")
            continue
        
        model.eval()
        model_preds = []
        with torch.no_grad():
            for images in tqdm(test_loader, desc=f"   Inferring", leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                model_preds.extend(probs.cpu().numpy())
        
        all_model_predictions.append(np.array(model_preds))

    if not all_model_predictions:
        print(f"🚨 '{group_config['group_name']}' 그룹에서 유효한 예측을 생성하지 못했습니다.")
        return None

    # 그룹 내 모델들의 가중 평균 앙상블
    print("\n⚖️ 그룹 내 가중 평균 앙상블을 수행합니다...")
    weights = compute_weights(group_config['models'])
    
    print("  - 계산된 가중치:")
    for i, model_info in enumerate(group_config['models']):
        print(f"    - {os.path.basename(model_info['path'])} (Loss: {model_info['val_log_loss']:.5f}): {weights[i]:.4f}")

    ensembled_predictions = np.average(np.array(all_model_predictions), axis=0, weights=weights)

    # 결과를 CSV로 저장
    try:
        submission_df = pd.read_csv(CFG['SAMPLE_SUBMISSION_PATH'])
        class_names = submission_df.columns[1:].tolist()
        pred_df = pd.DataFrame(ensembled_predictions, columns=class_names)
        submission_df[class_names] = pred_df[class_names].values
        
        output_path = group_config["output_csv_path"]
        submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ STAGE 1 완료. 결과 저장: {output_path}")
        return {
            "path": output_path,
            "val_log_loss": group_config["ensemble_log_loss"]
        }
    except Exception as e:
        print(f"🚨 CSV 파일 생성 중 오류 발생: {e}")
        return None

def run_stage2_csv_ensemble(csv_infos):
    """1단계에서 생성된 CSV들을 가중 평균하여 최종 제출 파일을 생성합니다."""
    print("\n" + "="*80)
    print(f"📦 STAGE 2: CSV 파일 앙상블 시작")
    print("="*80)

    try:
        dfs = [pd.read_csv(info["path"]) for info in csv_infos]
    except FileNotFoundError as e:
        print(f"🚨 앙상블할 CSV 파일을 찾을 수 없습니다: {e}")
        return
        
    # 가중치 계산
    print("⚖️ 2단계 앙상블 가중치를 계산합니다...")
    weights = compute_weights(csv_infos, loss_key='val_log_loss')
    
    print("  - 계산된 가중치:")
    for i, info in enumerate(csv_infos):
        print(f"    - {os.path.basename(info['path'])} (Group Loss: {info['val_log_loss']:.5f}): {weights[i]:.4f}")

    # 확률 값 추출 및 앙상블
    prob_cols = dfs[0].columns[1:]
    probs_stack = np.stack([df[prob_cols].values for df in dfs], axis=0)
    final_probs = np.average(probs_stack, axis=0, weights=weights)

    # 최종 제출 파일 생성
    final_submission_df = pd.read_csv(CFG['SAMPLE_SUBMISSION_PATH'])
    final_submission_df[prob_cols] = final_probs
    
    output_path = CFG['STAGE2_OUTPUT_NAME']
    final_submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 최종 앙상블 완료! 제출 파일 저장: {output_path}")

def main():
    print("🌟 2-Stage Weighted Ensemble Script를 시작합니다. 🌟")
    print(f"사용 디바이스: {device}")
    seed_everything(CFG['SEED'])

    try:
        with open(CFG['CLASS_NAMES_PATH'], 'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        print(f"클래스 정보 로드 완료 ({num_classes}개 클래스).")
    except FileNotFoundError:
        print(f"🚨 치명적 오류: {CFG['CLASS_NAMES_PATH']} 파일을 찾을 수 없습니다.")
        print("   먼저 train.py를 실행하여 클래스 파일을 생성해야 합니다.")
        return

    # --- STAGE 1 실행 ---
    stage1_results = []
    for group_config in CFG['STAGE1_GROUPS']:
        result_info = run_stage1_inference_ensemble(group_config, num_classes)
        if result_info:
            stage1_results.append(result_info)
    
    if len(stage1_results) != len(CFG['STAGE1_GROUPS']):
        print("\n🚨 1단계에서 일부 그룹 처리에 실패하여 2단계 앙상블을 진행할 수 없습니다.")
        return
        
    # --- STAGE 2 실행 ---
    run_stage2_csv_ensemble(stage1_results)

if __name__ == '__main__':
    main()
