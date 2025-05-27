import os
import sys
import random
import numpy as np
import torch
import importlib
import json
import albumentations as A
import torchvision.transforms as T

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)  # 터미널에도 출력
        self.log.write(message)       # 파일에도 기록

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# torch vision transform의 경우 저장을 위해 dictionary화 하는 로직이 따로 필요함
def serialize_torchvision_transform(transform):
    result = []
    for t in transform.transforms:
        t_dict = {
            "name": t.__class__.__name__,
            "params": {}
        }
        try:
            # Extract user-defined params only
            for k, v in vars(t).items():
                # Optionally filter non-serializable types
                try:
                    json.dumps(v)  # test if serializable
                    t_dict["params"][k] = v
                except TypeError:
                    t_dict["params"][k] = str(v)
        except Exception:
            t_dict["params"] = "Could not extract parameters"

        result.append(t_dict)
    return result

# transform 저장을 위해 추가
def save_transform(transform, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(transform, A.Compose):
        transform_dict = {
            "type": "albumentations",
            "transform": transform.to_dict()
        }
    elif isinstance(transform, T.Compose):
        transform_dict = {
            "type": "torchvision",
            "transform": serialize_torchvision_transform(transform)
        }
    else:
        raise ValueError(f"Unsupported transform type: {type(transform)}")

    with open(save_path, 'w') as f:
        json.dump(transform_dict, f, indent=4)

# 편한 로깅을 위해 만든 함수. 문자열로 모듈을 불러옴
def get_class_from_string(full_class_string):
    try:
        module_path, class_name = full_class_string.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {full_class_string}: {e}")

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_first_file_by_extension(directory, extension = '.pth'):
    # 주어진 디렉토리에서 파일 목록을 가져와서 필터링
    matched_files = [f for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extension.lower())]
    
    if not matched_files:
        return None  # 해당 확장자를 가진 파일이 없음

    # 사전순 정렬 후 첫 번째 파일 반환
    matched_files.sort()
    return os.path.join(directory, matched_files[0])