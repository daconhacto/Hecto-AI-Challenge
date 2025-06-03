import os
import sys
import random
import numpy as np
import torch
import importlib
import json
import albumentations as A
import torchvision.transforms as T
from collections import defaultdict

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


def find_class_groups_from_jsons(folder_path, start_epoch = 10):
    # 1. 클래스 간 연결 정보 추출
    class_graph = defaultdict(set)

    for file_name in os.listdir(folder_path):
        if int(file_name.split('_')[3]) < start_epoch:
            continue

        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for true_class, examples in data.items():
                    for ex in examples:
                        predicted_class = ex['model_answer']
                        # 양방향 연결로 그래프 구성
                        class_graph[true_class].add(predicted_class)
                        class_graph[predicted_class].add(true_class)

    # 2. 그래프 기반 그룹 찾기 (연결된 component 별로 묶기)
    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in class_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for cls in class_graph:
        if cls not in visited:
            group = []
            dfs(cls, group)
            groups.append(sorted(group))

    return groups

def get_total_wrong_groups(work_dir, start_epoch):
    wrong_examples = os.path.join(work_dir, 'wrong_examples')
    groups = find_class_groups_from_jsons(wrong_examples, start_epoch)
    with open(os.path.join(work_dir, 'groups.json'), 'w') as f:
        json.dump(groups, f)
    print(f'total wrong group saved to {os.path.join(work_dir, "groups.json")}')


# 각 클래스에 속한 샘플 인덱스 사전 생성
def build_class_index_map(samples):
    label_to_indices = defaultdict(list)
    for idx, sample in enumerate(samples):
        _, label = sample
        label_to_indices[label].append(idx)
    return label_to_indices

def convert_classname_groups_to_index_groups(groups, class_names):
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    index_groups = {}
    for i, group in enumerate(groups):
        index_groups[i] = [class_to_index[cls] for cls in group if cls in class_to_index]
    return index_groups