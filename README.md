# Hecto-AI-Challenge
최근 자동차 산업의 디지털 전환과 더불어, 다양한 차종을 빠르고 정확하게 인식하는 기술의 중요성이 커지고 있습니다. 특히 중고차 거래 플랫폼, 차량 관리 시스템, 자동 주차 및 보안 시스템 등 실생활에 밀접한 분야에서는 정확한 차종 분류가 핵심 기술로 떠오르고 있습니다.

이미지 기반 차종 인식 기술은 기존의 수작업 방식에 비해 높은 정확도와 효율성을 제공하며, 인공지능(AI)을 활용한 자동화 기술이 빠르게 발전하고 있습니다. 그중에서도 다양한 차종을 세밀하게 구분할 수 있는 능력은 실제 서비스 도입 시 차별화된 경쟁력을 좌우하는 요소로 작용합니다.

이에 따라, ‘HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회’는 실제 중고차 차량 이미지를 기반으로 한 차종 분류 AI 모델 개발을 주제로 개최됩니다.

# environment
- OS : Ubuntu 22.04.5 LTS
- CUDA Driver Version: 535.183.01
- CUDA Version: 12.2
- python 3.10.16

# installation
```
conda create -n hecto python=3.10.16
conda activate hecto
pip install -r requirements.txt
```
# 추론 파일 재현을 위한 code 실행



# How To Use
1. ../data 폴더에 train 학습파일 위치
2. /data_control/data_remove_v2.py 실행
3. /data_control/img_move_v1.py 실행

- train.py

4. convnext 모델 학습실행 
```
python train.py \
  --ROOT ../data/train \
  --WORK_DIR ../work_dir \
  --MODEL_NAME convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 \
  --N_FOLDS 5 \
  --IMG_SIZE 600
```
5. eva 모델 학습실행 
```
python train.py \
  --ROOT ../data/train \
  --WORK_DIR ../work_dir \
  --MODEL_NAME eva02_large_patch14_448.mim_in22k_ft_in1k \
  --N_FOLDS 5 \
  --IMG_SIZE 448
```

6. work_dir에 생긴 파일을 가지고 추론을 진행합니다.
- inference.py
```
python inference.py \
  --ROOT ../data/test \
  --SUBMISSION_FILE ../data/submission.csv \
  --WORK_DIR ../work_dir \
  --MODEL_PATH ../work_dir/best_model.pth \
  --BATCH_SIZE 64
```
