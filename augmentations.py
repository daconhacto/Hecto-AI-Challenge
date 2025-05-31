import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F


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
def apply_mosaic(images, labels, num_classes, p=0.5, grid_size=2, use_saliency=True):
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
                        lam = np.random.beta(1.0, 1.0)
                        bbx1, bby1, bbx2, bby2 = saliency_bbox(src_img, lam)
                        crop = src_img[:, bbx1:bbx2, bby1:bby2]
                        crop = F.interpolate(crop.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
                        new_image[:, y1:y2, x1:x2] = crop
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



def half_crop(image, mode=None):
    H, W, _ = image.shape
    if mode == 'top':
        return image[:H//2, :, :]
    elif mode == 'bottom':
        return image[H//2:, :, :]
    elif mode == 'left':
        return image[:, :W//2, :]
    elif mode == 'right':
        return image[:, W//2:, :]
    else:
        return image  # no crop

class CustomCropTransform:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, image, **kwargs):
        mode = self.mode if self.mode else random.choice(['top', 'bottom', 'left', 'right'])
        return half_crop(image, mode)