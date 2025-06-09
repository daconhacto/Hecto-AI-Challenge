import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from collections import defaultdict
from typing import Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchvision.transforms import v2

try:
    from diffusers import StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def get_default_train_transform_albu(img_size):
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Affine(translate_percent=(0.1, 0.1), scale=(0.9, 1.1), shear=10, rotate=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_default_val_transform_albu(img_size):
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_default_train_transform_torchvision(img_size):
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_default_val_transform_torchvision(img_size):
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    transforms.Compose([ # inf.py의 test_transform과 동일해야 함
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# 확률적으로 Mosaic 적용을 위한 함수
def apply_mosaic(images, labels, num_classes, p=1.0, grid_size=2, use_saliency=True):
    if random.random() < p:
        batch_size, channels, height, width = images.shape
        device = images.device

        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=num_classes).float().to(labels.device)

        mosaic_images = torch.zeros_like(images)
        mosaic_labels = torch.zeros_like(labels)

        for b_idx in range(batch_size):
            new_image = torch.zeros((channels, height, width), device=device)
            new_label = torch.zeros_like(labels[b_idx])
            cell_h, cell_w = height // grid_size, width // grid_size

            image_indices = [b_idx] + random.choices(list(range(batch_size)), k=(grid_size * grid_size) - 1)
            random.shuffle(image_indices)

            weights = []
            total_area = height * width

            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    src_idx = image_indices[idx]
                    src_img = images[src_idx]

                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w

                    if i > 0 and j > 0 and i < grid_size - 1 and j < grid_size - 1:
                        y1 = min(height - 1, max(0, y1 + random.randint(-cell_h // 4, cell_h // 4)))
                        y2 = min(height, max(y1 + 1, y2 + random.randint(-cell_h // 4, cell_h // 4)))
                        x1 = min(width - 1, max(0, x1 + random.randint(-cell_w // 4, cell_w // 4)))
                        x2 = min(width, max(x1 + 1, x2 + random.randint(-cell_w // 4, cell_w // 4)))

                    if use_saliency:
                        try:
                            lam = np.random.beta(1.0, 1.0)
                            bbx1, bby1, bbx2, bby2 = saliency_bbox(src_img, lam)
                            crop = src_img[:, bbx1:bbx2, bby1:bby2]
                            crop = F.interpolate(crop.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
                            new_image[:, y1:y2, x1:x2] = crop
                        except:
                            patch = src_img[:, y1:y2, x1:x2]
                            new_image[:, y1:y2, x1:x2] = patch
                    else:
                        patch = src_img[:, y1:y2, x1:x2]
                        new_image[:, y1:y2, x1:x2] = patch

                    cell_area = (y2 - y1) * (x2 - x1)
                    weight = cell_area / total_area
                    weights.append((src_idx, weight))
                    idx += 1

            for src_idx, weight in weights:
                new_label += labels[src_idx] * weight

            mosaic_images[b_idx] = new_image
            mosaic_labels[b_idx] = new_label

        return mosaic_images, mosaic_labels

    return images, labels


# cut out
def apply_cutout(images: torch.Tensor, mask_size: int = 32) -> torch.Tensor:
    """
    Efficient Cutout using vectorized torch operations.

    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        mask_size (int): Size of the square mask to cut out

    Returns:
        augmented images (torch.Tensor) : Batch of images [B, C, H, W]
    """
    B, C, H, W = images.shape
    cutout_images = images.clone()

    # 랜덤 중심 좌표 생성 [B]
    ys = torch.randint(0, H, (B,), device=images.device)
    xs = torch.randint(0, W, (B,), device=images.device)

    for i in range(B):
        y1 = max(0, ys[i] - mask_size // 2)
        y2 = min(H, ys[i] + mask_size // 2)
        x1 = max(0, xs[i] - mask_size // 2)
        x2 = min(W, xs[i] + mask_size // 2)
        cutout_images[i, :, y1:y2, x1:x2] = 0

    return cutout_images


class SaliencyMix:
    """
    SaliencyMix transform for a batch of images (Tensor BxCxHxW).
    원본 논문 구현을 단순화해 batch-tensor로 동작하도록 변형.
    """
    def __init__(self, num_classes, alpha=1.0, prob=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        self.sal_det = cv2.saliency.StaticSaliencyFineGrained_create()

    @torch.no_grad()
    def _get_sal_bbox(self, img_np, lam):
        """saliency map → 가장 salient patch 면적 ≈ lam"""
        H, W, _ = img_np.shape
        success, sal_map = self.sal_det.computeSaliency(img_np)
        sal_map = (sal_map * 255).astype("uint8")
        # threshold top-k pixels
        thresh = np.percentile(sal_map, 100 * (1 - lam))
        mask = sal_map >= thresh
        ys, xs = np.where(mask)
        if len(xs) == 0:   # fallback: 중앙 패치
            ys = np.arange(int(H * 0.25), int(H * 0.75))
            xs = np.arange(int(W * 0.25), int(W * 0.75))
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return x1, y1, x2, y2

    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels                       # 그냥 통과

        lam = np.random.beta(self.alpha, self.alpha)
        B, C, H, W = images.size()

        # ① 라벨을 one-hot(float)로 만들어 복사본을 사용
        labels = F.one_hot(labels, self.num_classes).to(images.dtype)

        rand_idx = torch.randperm(B, device=images.device)
        shuffled_images  = images[rand_idx]
        shuffled_labels  = labels[rand_idx]

        new_images = images.clone()
        new_labels = labels.clone()                     # 새 라벨 버퍼

        for i in range(B):
            donor = (shuffled_images[i]
                     .permute(1, 2, 0).mul(255)
                     .byte().cpu().numpy())
            x1, y1, x2, y2 = self._get_sal_bbox(donor[:, :, ::-1], lam)

            new_images[i, :, y1:y2, x1:x2] = shuffled_images[i, :, y1:y2, x1:x2]

            lam_i = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
            new_labels[i] = labels[i] * lam_i + shuffled_labels[i] * (1 - lam_i)

        return new_images, new_labels


def saliencymix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    num_candidates: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SaliencyMix augmentation.

    Args:
        images: [B, C, H, W], float32, 0~1 스케일
        labels: [B] (int) or [B, num_classes] (one-hot float)
        num_classes: 클래스 수
        alpha: Beta 분포 파라미터
        num_candidates: saliency 기반 후보 창 개수

    Returns:
        mixed_images: [B, C, H, W]
        mixed_labels: [B, num_classes] (soft labels)
    """
    B, C, H, W = images.shape
    device = images.device

    # 1) λ 샘플링 및 patch 크기 계산
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1 - lam)
    ph = int(H * cut_rat)
    pw = int(W * cut_rat)

    # 2) 레이블을 one-hot float32 로
    if labels.dim() == 1:
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    else:
        labels_onehot = labels.float().to(device)

    mixed_images = images.clone()
    mixed_labels = labels_onehot.clone()

    # 3) Sobel 커널 정의 (grayscale → saliency)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           dtype=torch.float32, device=device).view(1,1,3,3)

    # 4) 각 이미지마다 mix 수행
    for i in range(B):
        # a) 랜덤 대조 이미지 선택 (i 제외)
        j = random.choice([x for x in range(B) if x != i])

        # b) grayscale → sobel → saliency map
        gray = images[j].mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        sal_map = (gx.abs() + gy.abs()).squeeze(0).squeeze(0)    # [H,W]

        # c) 후보 위치 중 saliency 합 최대인 박스 찾기
        best_score = -1
        best_y, best_x = 0, 0
        for _ in range(num_candidates):
            y = np.random.randint(0, H - ph + 1)
            x = np.random.randint(0, W - pw + 1)
            score = float(sal_map[y:y+ph, x:x+pw].sum().item())
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

        # d) 이미지 patch 교체
        mixed_images[i, :, best_y:best_y+ph, best_x:best_x+pw] = images[j, :, best_y:best_y+ph, best_x:best_x+pw]

        # e) 레이블 믹싱 (면적 비율)
        area = ph * pw
        lam_adjusted = 1.0 - (area / (H * W))
        mixed_labels[i] = labels_onehot[i] * lam_adjusted + labels_onehot[j] * (1.0 - lam_adjusted)

    return mixed_images, mixed_labels


def half_mosaic(images, labels, class_names, similarity_map, num_classes, orientation='horizontal'):

    """
    배치 내 유사 클래스끼리 이미지를 절반씩 붙이고, 레이블도 0.5씩 혼합하여 증강.
    orientation: 'horizontal' 또는 'vertical'
    """
    B, C, H, W = images.shape
    device = images.device

    # 레이블이 원-핫 인코딩 되어있지 않다면 변환 (CrossEntropyLoss는 정수 레이블도 받으므로,
    # 이 함수를 호출하기 전에 또는 이 함수 내에서 일관되게 처리 필요)
    # mosaic_augmentation과 일관성을 위해 여기서 처리:
    if labels.ndim != 2 or labels.shape[1] != num_classes:
        original_int_labels = labels.clone() # class_names 조회용 원본 정수 레이블
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    else: # 이미 원-핫 인코딩된 float 레이블로 가정
        original_int_labels = torch.argmax(labels, dim=1) # class_names 조회용 정수 레이블 복원
        labels_one_hot = labels

    out_images = images.clone()
    # 초기 레이블은 원본 이미지의 원-핫 레이블로 설정 (혼합 대상이 없을 경우 대비)
    out_labels = labels_one_hot.clone()

    for i in range(B):
        # class_names, similarity_map 조회 시에는 정수 레이블 사용
        cls_i_name = class_names[original_int_labels[i].item()]
        candidates = similarity_map.get(cls_i_name, [])
        
        # 현재 배치 내에서 혼합할 다른 이미지 인덱스 찾기 (자기 자신 제외)
        candidate_indices_in_batch = [
            j_idx for j_idx in range(B)
            if i != j_idx and class_names[original_int_labels[j_idx].item()] in candidates
        ]
        
        if not candidate_indices_in_batch:
            # 혼합할 대상 이미지가 없으면 원본 이미지와 원본 레이블(원-핫) 사용
            continue # out_images[i]는 이미 원본, out_labels[i]도 이미 원본 레이블

        # 혼합할 이미지 j 랜덤 선택
        j = random.choice(candidate_indices_in_batch)
        
        img1, label1_one_hot = images[i], labels_one_hot[i]
        img2, label2_one_hot = images[j], labels_one_hot[j]
        
        # 이미지 절반씩 조합
        if orientation == 'vertical':
            top_half = img1[:, :H//2, :]
            bottom_half = img2[:, H//2:, :]
            out_images[i] = torch.cat([top_half, bottom_half], dim=1)
        else:  # horizontal
            left_half = img1[:, :, :W//2]
            right_half = img2[:, :, W//2:]
            out_images[i] = torch.cat([left_half, right_half], dim=2)
            
        # 레이블 0.5씩 혼합 (soft label 생성)
        out_labels[i] = 0.5 * label1_one_hot + 0.5 * label2_one_hot
        
    return out_images, out_labels

def mosaic_augmentation(images, labels, num_classes, grid_size=2):
    """
    배치 내의 이미지들을 사용하여 Mosaic 증강을 수행합니다.
    
    Args:
        images (torch.Tensor): 배치 이미지 텐서 [B, C, H, W]
        labels (torch.Tensor): 배치 레이블 텐서 [B, num_classes] - one-hot 인코딩 가정
        grid_size (int): Mosaic 그리드 크기 (기본값: 2x2 그리드)
    
    Returns:
        tuple: (mosaic_images, mosaic_labels) - 증강된 이미지와 레이블
    """
    batch_size, channels, height, width = images.shape
    device = images.device
    
    if labels.ndim != 2:
        labels = F.one_hot(labels, num_classes=num_classes).float().to(labels.device)
    
    # 원본 배치 크기 유지
    mosaic_images = torch.zeros_like(images)
    mosaic_labels = torch.zeros_like(labels)
    
    for b_idx in range(batch_size):
        # 각 이미지마다 Mosaic 적용
        new_image = torch.zeros((channels, height, width), device=device)
        new_label = torch.zeros_like(labels[b_idx])
        
        # grid_size x grid_size 그리드 생성
        cell_h, cell_w = height // grid_size, width // grid_size
        
        # 사용할 이미지 인덱스 선택 (현재 이미지 포함하여 무작위로)
        image_indices = [b_idx] + random.choices(list(range(batch_size)), k=(grid_size*grid_size)-1)
        random.shuffle(image_indices)
        
        # 가중치를 저장할 변수 (레이블 혼합에 사용)
        weights = []
        total_area = height * width
        
        # 그리드 채우기
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                src_idx = image_indices[idx]
                src_img = images[src_idx]
                
                # 셀 영역 계산
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                # 약간의 변동성을 위한 무작위 오프셋 (선택 사항)
                if i > 0 and j > 0 and i < grid_size-1 and j < grid_size-1:
                    y1 = min(height-1, max(0, y1 + random.randint(-cell_h//4, cell_h//4)))
                    y2 = min(height, max(y1+1, y2 + random.randint(-cell_h//4, cell_h//4)))
                    x1 = min(width-1, max(0, x1 + random.randint(-cell_w//4, cell_w//4)))
                    x2 = min(width, max(x1+1, x2 + random.randint(-cell_w//4, cell_w//4)))
                
                # 해당 셀에 이미지 할당
                cell_area = (y2 - y1) * (x2 - x1)
                new_image[:, y1:y2, x1:x2] = src_img[:, y1:y2, x1:x2]
                
                # 면적 가중치 계산
                weight = cell_area / total_area
                weights.append((src_idx, weight))
                
                idx += 1
        
        # 레이블 혼합
        for src_idx, weight in weights:
            new_label += labels[src_idx] * weight
        
        # 결과 저장
        mosaic_images[b_idx] = new_image
        mosaic_labels[b_idx] = new_label
    
    return mosaic_images, mosaic_labels


def random_half_mosaic(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    orientation: str = 'horizontal'
):
    """
    배치 내 임의의 다른 이미지와 절반씩 붙이는 Half-Mosaic.
    class 제약 없이 완전 랜덤 혼합.
    
    Args:
        images: [B, C, H, W]
        labels: [B] 또는 [B, num_classes]
        num_classes: 클래스 수
        orientation: 'horizontal' or 'vertical'
    Returns:
        mixed_images: [B, C, H, W]
        mixed_labels: [B, num_classes] (soft label)
    """
    B, C, H, W = images.shape
    device = images.device

    # labels → one-hot float32 로 변환
    if labels.dim() == 1:
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    else:
        labels_onehot = labels.float().to(device)

    out_images = images.clone()
    out_labels = labels_onehot.clone()

    for i in range(B):
        # 자기 자신 제외한 랜덤 인덱스 선택
        j = random.choice([k for k in range(B) if k != i])

        img1 = images[i]
        img2 = images[j]
        lab1 = labels_onehot[i]
        lab2 = labels_onehot[j]

        if orientation == 'vertical':
            top = img1[:, :H//2, :]
            bot = img2[:, H//2:, :]
            out_images[i] = torch.cat([top, bot], dim=1)
        else:  # horizontal
            left = img1[:, :, :W//2]
            right = img2[:, :, W//2:]
            out_images[i] = torch.cat([left, right], dim=2)

        # 레이블은 50:50 섞기
        out_labels[i] = 0.5 * lab1 + 0.5 * lab2

    return out_images, out_labels


# CustomCropTransform에서 이미지 비율에 따라 위/아래, 왼/옆 자르기 모드를 나누는 클래스
class CustomCropTransformConsiderRatio(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0, mode=None, consider_ratio=True):
        self.mode = mode
        self.consider_ratio = consider_ratio
        super().__init__(always_apply, p)

    def apply(self, image, **kwargs):
        h, w = image.shape[:2]
        if self.mode:
            mode = self.mode
        else:
            if self.consider_ratio:
                if h > w:
                    mode = random.choice(['top', 'bottom'])
                elif w > h:
                    mode = random.choice(['left', 'right'])
                else:
                    mode = random.choice(['top', 'bottom', 'left', 'right'])
            else:
                mode = self.mode if self.mode else random.choice(['top', 'bottom', 'left', 'right'])

        if mode == 'top':
            return image[:h//2, :, :]
        elif mode == 'bottom':
            return image[h//2:, :, :]
        elif mode == 'left':
            return image[:, :w//2, :]
        elif mode == 'right':
            return image[:, w//2:, :]
        else:
            return image  # no crop


def mosaic_selector(
    images, labels, 
    class_names, similarity_map,
    num_classes, CFG
):
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
    return images, labels


# refactoring 및 클래스 쌍 일단 저장용 함수
def get_similarity_map():
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
    similarity_map = defaultdict(set)
    for a, b in confusion_pairs:
        similarity_map[a].add(b)
        similarity_map[b].add(a)

    # 3) 최종 dict 형태로 변환
    similarity_map = {k: list(v) for k, v in similarity_map.items()}
    return similarity_map


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, old_target, num_classes, lam, smoothing=0.1, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(old_target, num_classes, on_value=on_value, off_value=off_value, device=device)
    
    return y1 * lam + y2 * (1. - lam)

def MaskMix(samples, flip_samples, alpha, mask_num, scale_, scale, targets, flip_targets, label_smoothing, num_classes):
    
    batch_size = samples.size(0)
    # total mask_num 
    token_count = mask_num ** 2
    ratio = np.random.beta(alpha, alpha) 
    #mask_ratio = [np.random.beta(alpha, alpha)  for i in range(batch_size)]
    mask_ratio = [ratio  for i in range(batch_size)]
    
    mask_count = [int(np.ceil(token_count * mask_ratio[i])) for i in range(batch_size)]
    
    mask_ratio = [mask_count[i]/token_count for i in range(batch_size)]
    
    mask_idx = [np.random.permutation(token_count)[:mask_count[i]] for i in range(batch_size)]
    mask = np.zeros((batch_size, token_count), dtype=int)
    for i in range(batch_size):
          mask[i][mask_idx[i]] = 1 
    mask = [mask[i].reshape((mask_num, mask_num)) for i in range(batch_size)]
    mask_ = [mask[i].repeat(scale_, axis=0).repeat(scale_, axis=1) for i in range(batch_size)]
    mask = [mask[i].repeat(scale, axis=0).repeat(scale, axis=1)  for i in range(batch_size)]
    mask = torch.from_numpy(np.array(mask)).to(samples.device)
    mask = mask.unsqueeze(1).repeat(1,3,1,1) #(128,224,224)->(128,1,224,224)->(128,3,224,224)
    mask = mask[:,:,:samples.shape[2],:samples.shape[2]]
    samples = samples * mask + flip_samples * (1-mask)

    mask_ratio = torch.Tensor(mask_ratio).to(samples.device)
    ratio = mask_ratio.unsqueeze(1).repeat(1, num_classes) #(128)->(128,1)->(128, num_classes)
    
    targets = mixup_target(targets, flip_targets, num_classes, ratio, label_smoothing, device=targets.device)

    mask_= np.array(mask_)
    return samples, targets, mask_, mask_ratio


class PuzzleMix:
    """
    PuzzleMix: Exploiting Saliency and Local Statistics for Optimal Mixup
    최신 2025년 기법 - saliency와 지역 통계를 활용한 최적화된 mixup
    OpenCV 호환성을 위해 Sobel 기반으로 구현
    """
    def __init__(self, num_classes, alpha=1.0, prob=0.5, block_num=2, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        self.block_num = block_num
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.neigh_size = neigh_size
        
    def _saliency_bbox(self, img, lam):
        """Sobel 기반 saliency bounding box 생성"""
        C, H, W = img.shape
        device = img.device
        
        # RGB to grayscale
        if C == 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        else:
            gray = img[0]
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        gray_4d = gray.unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(gray_4d, sobel_x, padding=1)
        grad_y = F.conv2d(gray_4d, sobel_y, padding=1)
        
        sal_map = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        # Top saliency 영역 찾기
        flat_sal = sal_map.flatten()
        threshold = torch.quantile(flat_sal, 1 - lam)
        mask = sal_map >= threshold
        
        ys, xs = torch.where(mask)
        
        if len(xs) == 0:
            # fallback to random box
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            return x1, y1, x2, y2
        
        # mask에서 bounding box 추출
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        
        return x1, y1, x2, y2
    
    def _local_statistics(self, img, x1, y1, x2, y2):
        """지역 통계 계산"""
        patch = img[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            return 0.0
        
        # 패치의 분산을 지역 복잡도로 사용
        return patch.var().item()
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        
        B, C, H, W = images.shape
        device = images.device
        
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        rand_idx = torch.randperm(B, device=device)
        images_shuffled = images[rand_idx]
        labels_shuffled = labels[rand_idx]
        
        new_images = images.clone()
        new_labels = labels.clone()
        
        for i in range(B):
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Saliency 기반 box 생성
            x1, y1, x2, y2 = self._saliency_bbox(images[i], lam)
            
            # 지역 통계 계산
            stat1 = self._local_statistics(images[i], x1, y1, x2, y2)
            stat2 = self._local_statistics(images_shuffled[i], x1, y1, x2, y2)
            
            # 통계 기반 mixing 가중치 조정
            stat_ratio = stat1 / (stat1 + stat2 + 1e-8)
            adjusted_lam = lam * stat_ratio
            
            # 패치 적용
            new_images[i, :, y1:y2, x1:x2] = images_shuffled[i, :, y1:y2, x1:x2]
            
            # 실제 면적 기반 lambda 계산
            actual_lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
            actual_lam = max(0.1, min(0.9, actual_lam))  # clipping
            
            new_labels[i] = actual_lam * labels[i] + (1 - actual_lam) * labels_shuffled[i]
        
        return new_images, new_labels


class CoMixup:
    """
    Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity
    2025년 최신 기법 - supermodular diversity를 활용한 joint mixup
    """
    def __init__(self, num_classes, alpha=1.0, prob=0.5, num_mix=4):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        self.num_mix = num_mix
    
    def _compute_diversity_matrix(self, features):
        """supermodular diversity 행렬 계산"""
        B = features.shape[0]
        # 코사인 유사도 계산
        features_norm = F.normalize(features.view(B, -1), dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        # diversity = 1 - similarity
        diversity = 1 - similarity
        return diversity
    
    def _optimal_pairing(self, diversity_matrix):
        """최적 페어링을 위한 헝가리안 알고리즘"""
        B = diversity_matrix.shape[0]
        # scipy를 위해 numpy로 변환
        div_np = diversity_matrix.cpu().numpy()
        # 최대 diversity를 위해 음수로 변환 (헝가리안은 최소값 찾음)
        cost_matrix = -div_np
        
        # 헝가리안 알고리즘
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        return list(zip(row_idx, col_idx))
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        
        B, C, H, W = images.shape
        device = images.device
        
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Feature extraction for diversity computation (간단한 pooling 사용)
        features = F.adaptive_avg_pool2d(images, (4, 4))
        
        # Diversity matrix 계산
        diversity_matrix = self._compute_diversity_matrix(features)
        
        # 최적 페어링
        pairs = self._optimal_pairing(diversity_matrix)
        
        new_images = images.clone()
        new_labels = labels.clone()
        
        for i, j in pairs[:B//2]:  # 절반만 mixup
            lam = np.random.beta(self.alpha, self.alpha)
            
            # 이미지 mixup
            new_images[i] = lam * images[i] + (1 - lam) * images[j]
            
            # 레이블 mixup
            new_labels[i] = lam * labels[i] + (1 - lam) * labels[j]
        
        return new_images, new_labels


class SnapMix:
    """
    SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data
    2025년 최신 - fine-grained classification을 위한 의미론적 비례 mixing
    간단한 gradient 기반 attention으로 구현 (CAM 대신)
    """
    def __init__(self, num_classes, alpha=5.0, prob=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        
    def _simple_attention_map(self, img):
        """간단한 attention 계산 (Sobel + variance 기반)"""
        C, H, W = img.shape
        device = img.device
        
        # RGB to grayscale
        if C == 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        else:
            gray = img[0]
        
        # Sobel + Local variance for attention
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        gray_4d = gray.unsqueeze(0).unsqueeze(0)
        
        # Edge detection
        grad_x = F.conv2d(gray_4d, sobel_x, padding=1)
        grad_y = F.conv2d(gray_4d, sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        # Local variance (texture)
        kernel_size = 5
        padding = kernel_size // 2
        unfold = F.unfold(gray_4d, kernel_size, padding=padding)
        unfold = unfold.view(1, kernel_size*kernel_size, H, W)
        local_mean = unfold.mean(dim=1, keepdim=True)
        local_var = ((unfold - local_mean)**2).mean(dim=1).squeeze()
        
        # Combine edge and texture
        attention = edge_map + local_var
        attention = F.normalize(attention.view(-1), dim=0).view(H, W)
        
        return attention.unsqueeze(0)  # [1, H, W]
    
    def _generate_snap_mask(self, img1, img2, lam):
        """SnapMix mask 생성"""
        H, W = img1.shape[-2:]
        
        # Attention 계산
        att1 = self._simple_attention_map(img1)
        att2 = self._simple_attention_map(img2)
        
        # 의미론적 비례에 따른 mask 생성
        combined_att = lam * att1 + (1 - lam) * att2
        
        # Threshold 기반 mask
        flat_att = combined_att.flatten()
        threshold = torch.quantile(flat_att, 1 - lam)
        mask = (combined_att >= threshold).float()
        
        return mask
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        
        B, C, H, W = images.shape
        device = images.device
        
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        rand_idx = torch.randperm(B, device=device)
        images_shuffled = images[rand_idx]
        labels_shuffled = labels[rand_idx]
        
        new_images = images.clone()
        new_labels = labels.clone()
        
        for i in range(B):
            lam = np.random.beta(self.alpha, self.alpha)
            
            # SnapMix mask 생성
            mask = self._generate_snap_mask(images[i], images_shuffled[i], lam)  # [1, H, W]
            
            # Mask 차원 맞추기: [1, H, W] -> [C, H, W]
            mask_expanded = mask.expand(C, H, W)
            new_images[i] = images[i] * mask_expanded + images_shuffled[i] * (1 - mask_expanded)
            
            # 실제 mixing ratio 계산
            actual_lam = mask.mean().item()
            actual_lam = max(0.1, min(0.9, actual_lam))
            
            new_labels[i] = actual_lam * labels[i] + (1 - actual_lam) * labels_shuffled[i]
        
        return new_images, new_labels


class DiffMix:
    """
    Diff-Mix: Enhanced Image Classification via Inter-class Image Mixup with Diffusion Model
    CVPR 2024 최신 기법 - diffusion model을 활용한 inter-class mixup
    
    Note: 실제 구현에서는 pretrained diffusion model이 필요하지만, 
    여기서는 간단한 노이즈 기반 approximation을 사용
    """
    def __init__(self, num_classes, alpha=1.0, prob=0.5, noise_steps=50):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        self.noise_steps = noise_steps
        
    def _add_noise(self, img, t):
        """간단한 가우시안 노이즈 추가 (실제 diffusion 모델 대신)"""
        noise = torch.randn_like(img)
        alpha = 1.0 - t / self.noise_steps
        # float를 tensor로 변환
        alpha_tensor = torch.tensor(alpha, device=img.device, dtype=img.dtype)
        sqrt_alpha = torch.sqrt(alpha_tensor)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_tensor)
        
        noisy_img = sqrt_alpha * img + sqrt_one_minus_alpha * noise
        return noisy_img
    
    def _denoise_mix(self, img1, img2, lam, t):
        """Diffusion-inspired mixing"""
        # 두 이미지에 다른 수준의 노이즈 추가
        noisy_img1 = self._add_noise(img1, t)
        noisy_img2 = self._add_noise(img2, t * 0.5)  # 다른 노이즈 레벨
        
        # 가중 평균
        mixed = lam * noisy_img1 + (1 - lam) * noisy_img2
        
        # 간단한 denoising (실제로는 diffusion model 필요)
        denoised = torch.clamp(mixed, 0, 1)
        
        return denoised
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        
        B, C, H, W = images.shape
        device = images.device
        
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 다른 클래스 간 페어링 우선
        unique_labels = torch.argmax(labels, dim=1)
        rand_idx = torch.randperm(B, device=device)
        
        # Inter-class pairing 시도
        for i in range(B):
            for j in range(i+1, B):
                if unique_labels[i] != unique_labels[rand_idx[j]]:
                    rand_idx[i] = rand_idx[j]
                    break
        
        images_shuffled = images[rand_idx]
        labels_shuffled = labels[rand_idx]
        
        new_images = images.clone()
        new_labels = labels.clone()
        
        for i in range(B):
            lam = np.random.beta(self.alpha, self.alpha)
            t = np.random.randint(1, self.noise_steps)
            
            # Diffusion-inspired mixing
            new_images[i] = self._denoise_mix(images[i], images_shuffled[i], lam, t)
            
            # 레이블 mixing
            new_labels[i] = lam * labels[i] + (1 - lam) * labels_shuffled[i]
        
        return new_images, new_labels


class DiffMixSDXL:
    """
    Enhanced Diff-Mix with SDXL Integration
    - 옵션 1: SDXL VAE encoder/decoder 활용한 latent space mixing
    - 옵션 2: SDXL 기반 controlled noise injection
    - 옵션 3: Fallback to improved Gaussian approximation
    """
    def __init__(self, num_classes, alpha=1.0, prob=0.5, noise_steps=50, 
                 use_sdxl=False, sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob
        self.noise_steps = noise_steps
        self.use_sdxl = use_sdxl and DIFFUSERS_AVAILABLE
        
        # SDXL 파이프라인 초기화 (옵션)
        if self.use_sdxl:
            try:
                print("Loading SDXL pipeline... (메모리 많이 사용됨 주의)")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    sdxl_model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                self.vae = self.pipe.vae
                print("SDXL pipeline loaded successfully!")
            except Exception as e:
                print(f"SDXL 로딩 실패: {e}")
                print("Fallback to Gaussian approximation")
                self.use_sdxl = False
        
    def _add_noise_gaussian(self, img, t):
        """개선된 가우시안 노이즈 (기존 방식 개선)"""
        noise = torch.randn_like(img)
        
        # 더 realistic한 노이즈 스케줄링
        beta_start = 0.0001
        beta_end = 0.02
        beta = beta_start + (beta_end - beta_start) * t / self.noise_steps
        
        alpha = 1.0 - beta
        alpha_bar = alpha ** t  # cumulative product approximation
        
        alpha_bar_tensor = torch.tensor(alpha_bar, device=img.device, dtype=img.dtype)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_tensor)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_tensor)
        
        noisy_img = sqrt_alpha_bar * img + sqrt_one_minus_alpha_bar * noise
        return torch.clamp(noisy_img, 0, 1)
    
    def _add_noise_sdxl(self, img, t):
        """SDXL VAE를 활용한 노이즈 추가"""
        if not self.use_sdxl:
            return self._add_noise_gaussian(img, t)
        
        try:
            with torch.no_grad():
                # VAE encoder로 latent space로 변환
                latents = self.vae.encode(img).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Latent space에서 노이즈 추가
                noise = torch.randn_like(latents)
                alpha = 1.0 - t / self.noise_steps
                alpha_tensor = torch.tensor(alpha, device=latents.device, dtype=latents.dtype)
                sqrt_alpha = torch.sqrt(alpha_tensor)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_tensor)
                
                noisy_latents = sqrt_alpha * latents + sqrt_one_minus_alpha * noise
                
                # VAE decoder로 다시 이미지 공간으로
                noisy_latents = noisy_latents / self.vae.config.scaling_factor
                decoded = self.vae.decode(noisy_latents).sample
                
                return torch.clamp(decoded, 0, 1)
        except Exception as e:
            print(f"SDXL 노이즈 추가 실패, Gaussian fallback: {e}")
            return self._add_noise_gaussian(img, t)
    
    def _denoise_mix(self, img1, img2, lam, t):
        """향상된 Diffusion-inspired mixing"""
        if self.use_sdxl:
            # SDXL 기반 노이즈 추가
            noisy_img1 = self._add_noise_sdxl(img1.unsqueeze(0), t).squeeze(0)
            noisy_img2 = self._add_noise_sdxl(img2.unsqueeze(0), int(t * 0.7)).squeeze(0)
        else:
            # 개선된 가우시안 노이즈
            noisy_img1 = self._add_noise_gaussian(img1, t)
            noisy_img2 = self._add_noise_gaussian(img2, int(t * 0.7))
        
        # 가중 평균 with adaptive lambda
        adaptive_lam = lam * (1.0 - t / self.noise_steps * 0.3)  # 노이즈 레벨에 따른 조정
        mixed = adaptive_lam * noisy_img1 + (1 - adaptive_lam) * noisy_img2
        
        # 자동차 특화: edge-preserving denoising
        denoised = self._edge_preserving_denoise(mixed)
        
        return torch.clamp(denoised, 0, 1)
    
    def _edge_preserving_denoise(self, img):
        """자동차 엣지 보존 디노이징"""
        # 간단한 bilateral filter 효과 근사
        kernel_size = 3
        padding = kernel_size // 2
        
        # 가우시안 블러 적용
        blurred = F.avg_pool2d(
            img.unsqueeze(0), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        ).squeeze(0)
        
        # 원본과 블러의 가중 평균 (엣지 보존 효과)
        edge_preserved = 0.7 * img + 0.3 * blurred
        return edge_preserved
    
    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        
        B, C, H, W = images.shape
        device = images.device
        
        if labels.ndim != 2:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Inter-class pairing (자동차 연도별 구분에 중요)
        unique_labels = torch.argmax(labels, dim=1)
        rand_idx = torch.randperm(B, device=device)
        
        # 다른 클래스끼리 우선 페어링
        for i in range(B):
            for j in range(B):
                if unique_labels[i] != unique_labels[rand_idx[j]]:
                    rand_idx[i] = rand_idx[j]
                    break
        
        images_shuffled = images[rand_idx]
        labels_shuffled = labels[rand_idx]
        
        new_images = images.clone()
        new_labels = labels.clone()
        
        for i in range(B):
            # 자동차 특화: 더 conservative한 lambda 값
            lam = np.random.beta(max(self.alpha, 0.8), max(self.alpha, 0.8))
            lam = max(lam, 0.6)  # 최소 0.6 이상 보장 (원본 특징 보존)
            
            t = np.random.randint(5, min(self.noise_steps, 30))  # 적당한 노이즈 레벨
            
            # Enhanced diffusion mixing
            new_images[i] = self._denoise_mix(images[i], images_shuffled[i], lam, t)
            
            # 레이블 mixing
            new_labels[i] = lam * labels[i] + (1 - lam) * labels_shuffled[i]
        
        return new_images, new_labels


# 개선된 자동차 특화 크롭
def enhanced_half_crop(image, mode=None):
    """자동차 특화 크롭"""
    H, W, _ = image.shape
    
    if mode == 'top':
        # 자동차 상단부 (루프, 윈도우 등)
        return image[:int(H*0.6), :, :]  # 60%만 크롭
    elif mode == 'bottom':
        # 자동차 하단부 (범퍼, 그릴 등)
        return image[int(H*0.4):, :, :]
    elif mode == 'left':
        return image[:, :int(W*0.6), :]
    elif mode == 'right':
        return image[:, int(W*0.4):, :]
    elif mode == 'center':
        # 중앙부 크롭 (자동차 핵심 부분)
        h_start, h_end = int(H*0.2), int(H*0.8)
        w_start, w_end = int(W*0.2), int(W*0.8)
        return image[h_start:h_end, w_start:w_end, :]
    else:
        return image


class EnhancedCustomCropTransform:
    def __init__(self, mode=None, p=1.0):
        self.mode = mode
        self.p = p

    def __call__(self, image, **kwargs):
        if random.random() < self.p:
            modes = ['top', 'bottom', 'left', 'right', 'center']
            mode = self.mode if self.mode else random.choice(modes)
            return enhanced_half_crop(image, mode)
        return image


class AdvancedAugmentationPipeline:
    """
    2025년 최신 augmentation 기법들을 조합한 파이프라인
    차량 분류에 특화된 설정
    """
    def __init__(self, num_classes, config=None):
        self.num_classes = num_classes
        
        # 기본 설정 (차량 분류 최적화)
        default_config = {
            'puzzlemix': {'prob': 0.3, 'alpha': 1.0},
            'comixup': {'prob': 0.2, 'alpha': 1.0},
            'snapmix': {'prob': 0.3, 'alpha': 5.0},
            'diffmix': {'prob': 0.2, 'alpha': 1.0},
            'saliencymix': {'prob': 0.25, 'alpha': 1.0},
            'mosaic': {'prob': 0.2, 'grid_size': 2},
            'cutout': {'prob': 0.3, 'mask_size': 32}
        }
        
        self.config = config if config else default_config
        
        # 각 기법 초기화
        self.puzzlemix = PuzzleMix(num_classes, **self.config['puzzlemix'])
        self.comixup = CoMixup(num_classes, **self.config['comixup'])
        self.snapmix = SnapMix(num_classes, **self.config['snapmix'])
        self.diffmix = DiffMix(num_classes, **self.config['diffmix'])
        self.saliencymix = SaliencyMix(num_classes, **self.config['saliencymix'])
    
    def __call__(self, images, labels):
        """순차적으로 여러 기법 적용 (확률적)"""
        # 가장 효과적인 기법들을 우선 적용
        
        # 1. SnapMix (fine-grained에 효과적) - 확률적 적용
        if random.random() < 0.3:  # 30% 확률로만 적용
            images, labels = self.snapmix(images, labels)
        
        # 2. PuzzleMix (saliency + local statistics) - 확률적 적용
        if random.random() < 0.3:  # 30% 확률로만 적용
            images, labels = self.puzzlemix(images, labels)
        
        # 3. Cutout (간단하지만 효과적)
        if random.random() < self.config['cutout']['prob']:
            images = apply_cutout(images, self.config['cutout']['mask_size'])
        
        # 4. DiffMix (inter-class mixing) - 확률적 적용
        if random.random() < 0.2:  # 20% 확률로만 적용
            images, labels = self.diffmix(images, labels)
        
        # 5. Co-Mixup (diversity 기반) - 확률적 적용
        if random.random() < 0.2:  # 20% 확률로만 적용
            images, labels = self.comixup(images, labels)
        
        # 6. Mosaic (가끔)
        if random.random() < self.config['mosaic']['prob']:
            images, labels = apply_mosaic(
                images, labels, self.num_classes, 
                p=1.0,  # 이미 확률 체크했으므로 100% 적용
                grid_size=self.config['mosaic']['grid_size']
            )
        
        return images, labels


# Helper functions
def create_diffmix_transform(num_classes, use_sdxl=False):
    """
    DiffMix 변환 생성
    
    Args:
        num_classes: 클래스 수
        use_sdxl: SDXL 사용 여부 (메모리 많이 사용함 주의)
    """
    return DiffMixSDXL(
        num_classes=num_classes,
        alpha=1.2,  # 자동차용 조정값
        prob=0.5,
        noise_steps=50,
        use_sdxl=use_sdxl
    )

class RandomMixAugmentation:
    def __init__(self, CFG, num_classes):
        self.CFG =CFG
        target_augmentations = CFG['ALL_AUGMENTATIONS']
        self.selected_augmentations = [i for i in target_augmentations if CFG[i]['enable']] + CFG['NONE_AUGMENTATION_LIST']
        
        print(f"현재 TARGETs : {target_augmentations}")
        print(f"선택된 AUGMENTATIONS : {self.selected_augmentations}")
        # cutmix or mixup transform settings
        if CFG['CUTMIX']['enable'] and CFG["MIXUP"]['enable']:
            cutmix = v2.CutMix(num_classes=num_classes, **CFG['CUTMIX']['params'])
            mixup = v2.MixUp(num_classes=num_classes, **CFG['MIXUP']['params'])
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            print("매 배치마다 CUTMIX와 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
        elif CFG['CUTMIX']['enable']:
            self.cutmix_or_mixup = v2.CutMix(num_classes=num_classes, **CFG['CUTMIX']['params'])
            print("매 배치마다 CUTMIX를 랜덤하게 적용합니다. CFG를 확인하세요.")
        elif CFG["MIXUP"]['enable']:
            self.cutmix_or_mixup = v2.MixUp(num_classes=num_classes, **CFG['MIXUP']['params'])
            print("매 배치마다 MIXUP을 랜덤하게 적용합니다. CFG를 확인하세요.")
        else:
            self.cutmix_or_mixup = None
    
    def forward(self, images, labels):
        if self.selected_augmentations:
            choice = random.choice(self.selected_augmentations)
            if choice == "NONE":
                choice = None
        else:
            choice = None
        
        # cutout을 위해 추가
        if self.CFG['CUTOUT']['enable'] and choice == 'CUTOUT':
            images = apply_cutout(images, **self.CFG['CUTOUT']['params'])
        
        # cutmix mixup을 위해 추가
        if self.cutmix_or_mixup:
            images, labels = self.cutmix_or_mixup(images, labels)
        
        # MOSAIC을 위해 추가
        if self.CFG['MOSAIC']['enable'] and (choice == 'MOSAIC'):
            images, labels = apply_mosaic(images, labels, self.num_classes, **self.CFG['MOSAIC']['params'])
        
        # SaliencyMix를 위해 추가
        if choice == 'SALIENCYMIX' and self.CFG['SALIENCYMIX']['enable']:
            images, labels = saliencymix(images, labels, self.num_classes, **self.CFG['SALIENCYMIX']['params'])
        return images, labels