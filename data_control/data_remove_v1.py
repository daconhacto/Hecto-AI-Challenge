import os
import glob

# 삭제할 파일 목록 설정
files_to_delete = {
    "7시리즈_G11_2016_2018": [40],
    "GLE_클래스_W167_2019_2024": [68],
    "SM7_뉴아트_2008_2011": [53],
    "더_기아_레이_EV_2024_2025": [78],
    "디_올뉴그랜저_2023_2025": [39],
    "아반떼_N_2022_2023": [35, 64],
    "아베오_2012_2016": [52]
}

# 기본 경로 설정 (실제 경로로 수정해야 함)
base_dir = "/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/train_renovate_v1"

# 각 폴더에서 해당 번호의 파일 삭제
for folder, numbers in files_to_delete.items():
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.exists(folder_path):
        for number in numbers:
            # 파일명 패턴 생성 (예: GLE_클래스_W167_2019_2024_0068.jpg)
            file_pattern = f"{folder}_{number:04d}.jpg"
            file_path = os.path.join(folder_path, file_pattern)
            
            # glob을 사용하여 패턴에 맞는 파일 찾기 (대소문자 구분 없이)
            matching_files = glob.glob(os.path.join(folder_path, f"{folder}_*{number:04d}*.jpg"))
            
            if matching_files:
                for file in matching_files:
                    print(f"삭제: {file}")
                    # 실제 삭제를 하려면 아래 주석을 해제하세요
                    # os.remove(file)
            else:
                print(f"파일을 찾을 수 없음: {file_path}")
    else:
        print(f"폴더를 찾을 수 없음: {folder_path}")