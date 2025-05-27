import os
import sys
import random
import numpy as np
import torch

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