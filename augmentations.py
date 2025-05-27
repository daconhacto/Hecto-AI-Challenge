import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F


# 확률적으로 Mosaic 적용을 위한 함수
def apply_mosaic(images, labels, num_classes, p=0.5, grid_size=2):
    """
    주어진 확률로 배치에 Mosaic 증강을 적용합니다.
    
    Args:
        images (torch.Tensor): 배치 이미지 텐서
        labels (torch.Tensor): 배치 레이블 텐서
        p (float): Mosaic을 적용할 확률 (0.0 ~ 1.0)
        
    Returns:
        tuple: (images, labels) - 원본 또는 증강된 이미지와 레이블
    """
    if random.random() < p:
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