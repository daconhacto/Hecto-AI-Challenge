import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image # TestCustomImageDataset에서 사용될 수 있음

# train.py에서 필요한 클래스 및 함수를 가져온다고 가정합니다.
# 실제 환경에서는 train.py의 정확한 경로 및 내용을 확인해야 합니다.
# 여기서는 CustomTimmModel, TestCustomImageDataset, seed_everything, TRAIN_CFG가 있다고 가정합니다.
# 만약 train.py에 해당 정의가 없다면, 이 스크립트 내에 직접 정의하거나 올바르게 import 해야 합니다.

# ---- train.py로부터 가져와야 할 (또는 이 파일에 정의해야 할) 요소들 ----
# 예시:
class CustomTimmModel(torch.nn.Module): # train.py의 CustomTimmModel 정의와 동일해야 함
    def __init__(self, model_name, num_classes_to_predict, pretrained=True):
        super(CustomTimmModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.feature_dim = self.backbone.num_features
        self.head = torch.nn.Linear(self.feature_dim, num_classes_to_predict)
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class TestCustomImageDataset(Dataset): # train.py의 TestCustomImageDataset 정의와 동일해야 함
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
            # 예시 TRAIN_CFG가 없으므로 기본값 사용 또는 CFG_ENS에서 IMG_SIZE 가져오도록 수정 필요
            return torch.zeros((3, CFG_ENS.get('DEFAULT_IMG_SIZE', 512), CFG_ENS.get('DEFAULT_IMG_SIZE', 512)))
        if self.transform:
            image = self.transform(image)
        return image

def seed_everything(seed): # train.py의 seed_everything 정의와 동일해야 함
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# timm 라이브러리가 필요합니다. pip install timm
import timm
# -----------------------------------------------------------------------

# 현재 스크립트용 설정
CFG_ENS = {
    'MODELS_TO_ENSEMBLE': [
        # 중요: 아래 'val_log_loss'는 각 모델의 실제 검증 Log Loss 값으로 채워야 합니다!
       # {'path': '/data2/project/2025winter/tjrgus357/hacto/check_point/1114.pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 640, 'val_log_loss': 0.1114},
        #{'path': '/data2/project/2025winter/tjrgus357/hacto/check_point/EVA_large(12139)_fold1.pth', 'name': 'eva02_large_patch14_448.mim_m38m_ft_in1k', 'img_size': 448, 'val_log_loss': 0.12139},
        # {'path': '/home/tjrgus357/hecto_original/check_point/Salinecy(13354)_fold1.pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 512, 'val_log_loss': 0.13354}, 
        {'path': '/data2/project/2025winter/tjrgus357/hacto/check_point/1089.pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 640, 'val_log_loss': 0.1089}, 


        {'path': '/data2/project/2025winter/tjrgus357/hacto/check_point/conv_large(0.09476).pth', 'name': 'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384', 'img_size': 600, 'val_log_loss': 0.09476},
        #{'path': '/home/tjrgus357/hecto_original/check_point/EVA_large(12139)_fold1.pth', 'name': 'eva02_large_patch14_448.mim_m38m_ft_in1k', 'img_size': 448, 'val_log_loss': 0.12139},
        #{'path': '/home/tjrgus357/hecto_original/check_point/Salinecy(13354)_fold1.pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 512, 'val_log_loss': 0.13354}, 

        #{'path': '/home/tjrgus357/hecto_original/check_point/convnext(1218)_fold1.pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 512, 'val_log_loss': 0.12188},
        # {'path': '/home/tjrgus357/hecto_original/check_point/EVA_large(12139)_fold1.pth', 'name': 'eva02_large_patch14_448.mim_m38m_ft_in1k', 'img_size': 448, 'val_log_loss': 0.12139},
        # {'path': '/data2/project/2025winter/tjrgus357/hacto/check_point/img_size_640(11684).pth', 'name': 'convnext_base.fb_in22k_ft_in1k_384', 'img_size': 640, 'val_log_loss': 0.11684}, 
 
    ],
    'BATCH_SIZE': 16, # 추론 시 배치 크기 조절 가능
    'SEED': 42,
    'CLASS_NAMES_PATH': './class_names.json', # train.py 실행 후 생성된 class_names.json 경로
    'TEST_ROOT': '../test',
    'SAMPLE_SUBMISSION_PATH': '../sample_submission.csv',
    'DEFAULT_IMG_SIZE': 512 # TestCustomImageDataset의 더미 이미지 생성 시 사용
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_predictions_for_single_model(model_path, model_architecture_name, num_classes, test_loader_for_model):
    """지정된 모델 경로와 아키텍처로 모델을 로드하고 예측 확률을 반환합니다."""
    model = CustomTimmModel(model_name=model_architecture_name, num_classes_to_predict=num_classes).to(device)
    try:
        # CPU로 모델 로드 후 GPU로 보내는 것이 메모리 문제 발생 시 도움이 될 수 있음
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device) # 그 다음 GPU로
        print(f"Successfully loaded model: {os.path.basename(model_path)} (Arch: {model_architecture_name})")
    except Exception as e:
        print(f"Error loading model {model_path} (Arch: {model_architecture_name}): {e}")
        return None
    
    model.eval()
    model_predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader_for_model, desc=f"Inferring with {os.path.basename(model_path)}", leave=False):
            images = images.to(device)
            outputs = model(images) # 로짓
            probs = F.softmax(outputs, dim=1) # 확률
            model_predictions.extend(probs.cpu().numpy())
    return np.array(model_predictions)

def ensemble_main():
    print("Starting Weighted Averaging Ensemble Inference...")
    print(f"Using device: {device}")
    seed_everything(CFG_ENS['SEED'])

    try:
        with open(CFG_ENS['CLASS_NAMES_PATH'], 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Error: {CFG_ENS['CLASS_NAMES_PATH']} not found. Please run train.py first to generate it.")
        return
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.") # 클래스 이름은 출력하지 않음 (너무 길 수 있음)

    if not CFG_ENS['MODELS_TO_ENSEMBLE']:
        print("No models specified in 'MODELS_TO_ENSEMBLE'. Please populate it.")
        return

    print(f"\nModels to be ensembled ({len(CFG_ENS['MODELS_TO_ENSEMBLE'])}):")
    for model_info in CFG_ENS['MODELS_TO_ENSEMBLE']:
        print(f"  - Path: {os.path.basename(model_info['path'])}, Arch: {model_info['name']}, ImgSize: {model_info['img_size']}, ValLogLoss: {model_info.get('val_log_loss', 'N/A')}")

    all_model_fold_predictions = []
    successfully_processed_model_infos = [] # 성공적으로 처리된 모델의 정보만 저장
    
    current_img_size_for_loader = -1
    test_loader = None

    for model_info in CFG_ENS['MODELS_TO_ENSEMBLE']:
        model_path = model_info['path']
        model_architecture_name = model_info['name']
        model_img_size = model_info['img_size']
        
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist. Skipping this model.")
            continue
        
        # val_log_loss가 없으면 이 모델은 가중 평균에서 제외 (또는 기본 가중치 부여)
        if 'val_log_loss' not in model_info or model_info['val_log_loss'] is None:
            print(f"Warning: 'val_log_loss' not provided for {os.path.basename(model_path)}. This model might be skipped in weighted average or given default weight.")
            # 우선은 이런 모델도 예측은 하도록 하고, 가중치 계산 시 제외하거나 처리
            # 여기서는 일단 예측은 수행하고, 가중치 계산 시점에 val_log_loss가 있는 모델만 사용

        if test_loader is None or current_img_size_for_loader != model_img_size:
            print(f"\nCreating/Recreating test_loader for img_size: {model_img_size}...")
            current_test_transform = get_test_transform(model_img_size)
            test_dataset = TestCustomImageDataset(CFG_ENS['TEST_ROOT'], transform=current_test_transform)
            if not test_dataset.samples:
                print(f"No images found in {CFG_ENS['TEST_ROOT']}. Cannot create DataLoader for img_size {model_img_size}.")
                continue 
            test_loader = DataLoader(test_dataset, batch_size=CFG_ENS['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
            current_img_size_for_loader = model_img_size
            print(f"DataLoader ready for {len(test_dataset)} test images.")
        
        fold_predictions = get_predictions_for_single_model(
            model_path, 
            model_architecture_name,
            num_classes,
            test_loader
        )
        if fold_predictions is not None:
            all_model_fold_predictions.append(fold_predictions)
            successfully_processed_model_infos.append(model_info) # 예측 성공 시에만 추가
        else:
            print(f"Could not get predictions for {os.path.basename(model_path)}. Excluding from ensemble.")

    if not all_model_fold_predictions:
        print("No predictions were generated from any model. Cannot proceed with ensemble.")
        return

    # --- 가중치 계산 및 적용 ---
    weights_calculated = []
    valid_models_for_weighting_indices = [] # 가중치 계산에 실제 사용될 모델들의 인덱스

    for idx, model_info in enumerate(successfully_processed_model_infos):
        val_loss = model_info.get('val_log_loss')
        if val_loss is not None and val_loss > 0: # LogLoss는 0보다 커야 함
            # 더 좋은 성능(낮은 LogLoss)에 더 높은 가중치를 부여하는 방법:
            # 1. 1 / LogLoss
            # 2. max_log_loss - LogLoss (모든 LogLoss가 양수이고, max_log_loss보다 작을 때)
            # 3. exp(-k * LogLoss) 등
            # 여기서는 1 / LogLoss 사용 (간단하고 직관적)
            weight_score = 1.0 / val_loss 
            weights_calculated.append(weight_score)
            valid_models_for_weighting_indices.append(idx) # 이 모델의 예측을 가중 평균에 사용
        else:
            print(f"Warning: Skipping {os.path.basename(model_info['path'])} from weighted average due to missing or invalid val_log_loss ({val_loss}).")

    if not weights_calculated or not valid_models_for_weighting_indices:
        print("No valid weights could be calculated (e.g., all models missing val_log_loss or val_log_loss <= 0). Falling back to simple averaging for all processed models.")
        ensembled_predictions = np.mean(np.array(all_model_fold_predictions), axis=0)
    else:
        # 가중치 계산에 사용된 모델들의 예측만 필터링
        predictions_for_weighting = [all_model_fold_predictions[i] for i in valid_models_for_weighting_indices]
        
        total_weight_score = sum(weights_calculated)
        if total_weight_score == 0: # 모든 유효 가중치가 0인 극단적 경우 방지
            print("Sum of valid weight scores is zero. Falling back to simple averaging for all processed models.")
            ensembled_predictions = np.mean(np.array(all_model_fold_predictions), axis=0)
        else:
            normalized_weights = [w / total_weight_score for w in weights_calculated]
            
            print(f"\nApplying Weighted Averaging using {len(normalized_weights)} models with valid val_log_loss:")
            for i, model_idx in enumerate(valid_models_for_weighting_indices):
                model_info = successfully_processed_model_infos[model_idx]
                print(f"  - {os.path.basename(model_info['path'])} (ValLoss: {model_info['val_log_loss']:.4f}): Weight = {normalized_weights[i]:.4f}")
            
            ensembled_predictions = np.average(np.array(predictions_for_weighting), axis=0, weights=normalized_weights)
            print(f"\nSuccessfully applied weighted averaging.")

    # 제출 파일 생성
    pred_df = pd.DataFrame(ensembled_predictions, columns=class_names)

    try:
        submission_df = pd.read_csv(CFG_ENS['SAMPLE_SUBMISSION_PATH'], encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Sample submission file {CFG_ENS['SAMPLE_SUBMISSION_PATH']} not found.")
        return

    class_columns_in_submission = submission_df.columns[1:]
    try:
        submission_df[class_columns_in_submission] = pred_df[class_columns_in_submission].values
    except KeyError as e:
        print(f"KeyError: Mismatch between submission file columns and predicted class names: {e}")
        print(f"Please ensure your class_names.json (derived from training) matches the submission format.")
        return
    except ValueError as e:
        print(f"ValueError during submission assignment (likely shape mismatch - number of test images or classes): {e}")
        return
    
    output_submission_filename = f'weight_emsemble_v8.csv'
    submission_df.to_csv(output_submission_filename, index=False, encoding='utf-8-sig')
    print(f"🎉 Weighted ensemble submission file created successfully: {output_submission_filename}")

if __name__ == '__main__':
    # 이 스크립트를 실행하기 전에 CFG_ENS['MODELS_TO_ENSEMBLE'] 안의
    # 각 모델 딕셔너리에 'val_log_loss' 키와 실제 검증 LogLoss 값을 추가해야 합니다.
    ensemble_main()