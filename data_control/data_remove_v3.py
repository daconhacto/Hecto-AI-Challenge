import os
import shutil

# --- 설정 ---
# 기준이 되는 학습 데이터 루트 폴더 경로
BASE_TRAIN_DIR = '/project/ahnailab/jys0207/CP/tjrgus5/train_renovate_v3' # 고객님의 CFG["ROOT"] 값

# 남겨야 할 클래스 이름들 (confusion_pairs 리스트에서 추출)
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
    ("라브4_5세대_2019_2024", "RAV4_5세대_2019_2024"), # 동일 클래스명 쌍도 포함될 수 있음 (set으로 처리)
    ("레인지로버_이보크_2세대_2023_2024", "레인지로버_이보크_2세대_2020_2022"),
    ("스팅어_마이스터_2021_2023", "스팅어_2018_2020"),
    ("3시리즈_GT_F34_2014_2021", "4시리즈_F32_2014_2020"),
    ("GLE_클래스_W166_2016_2018", "4시리즈_G22_2024_2025"), # 이전에 나온 클래스도 포함될 수 있음
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

# confusion_pairs에 있는 모든 클래스 이름을 중복 없이 set으로 추출
classes_to_keep = set()
for pair in confusion_pairs:
    classes_to_keep.add(pair[0])
    classes_to_keep.add(pair[1])

# --- 실행 함수 ---
def delete_images_from_other_folders(base_dir, keep_folders_set):
    """
    base_dir 내의 모든 하위 폴더를 확인하여,
    keep_folders_set에 포함되지 않은 폴더 내의 모든 이미지 파일을 삭제합니다.
    폴더 자체는 삭제하지 않고, 이미지 파일만 삭제합니다.
    """
    deleted_files_count = 0
    kept_folders_count = 0
    processed_folders_count = 0

    print(f"기준 디렉토리: {base_dir}")
    print(f"이미지를 보존할 폴더 목록 ({len(keep_folders_set)}개): {sorted(list(keep_folders_set))}\n")

    if not os.path.isdir(base_dir):
        print(f"❌ 오류: 기준 디렉토리 '{base_dir}'를 찾을 수 없습니다.")
        return

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            processed_folders_count += 1
            if folder_name in keep_folders_set:
                print(f"🟢 보존 대상 폴더: '{folder_name}' (이미지 삭제 안 함)")
                kept_folders_count += 1
            else:
                print(f"🟡 삭제 대상 폴더 (내부 이미지): '{folder_name}'")
                current_folder_deleted_count = 0
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            os.remove(file_path)
                            # print(f"  - 삭제됨: {filename}") # 너무 많은 로그를 방지하기 위해 주석 처리
                            deleted_files_count += 1
                            current_folder_deleted_count +=1
                        except Exception as e:
                            print(f"  - ❗️ 오류: '{filename}' 삭제 중 오류 발생 - {e}")
                if current_folder_deleted_count > 0:
                    print(f"  ➡️ '{folder_name}' 폴더에서 {current_folder_deleted_count}개의 이미지 삭제 완료.")
                else:
                    print(f"  ➡️ '{folder_name}' 폴더에 삭제할 이미지 파일이 없습니다.")
    
    print(f"\n--- 작업 요약 ---")
    print(f"총 처리된 폴더 수: {processed_folders_count}")
    print(f"이미지 보존된 폴더 수: {kept_folders_count}")
    print(f"내부 이미지가 삭제된 폴더 수: {processed_folders_count - kept_folders_count}")
    print(f"총 삭제된 이미지 파일 수: {deleted_files_count}")
    print(f"✅ 작업 완료.")

# --- 실행 ---
if __name__ == "__main__":
    print("="*50)
    print("이미지 삭제 스크립트를 시작합니다.")
    print("이 스크립트는 지정된 '보존 대상 폴더' 외의 모든 폴더에서 이미지 파일을 삭제합니다.")
    print("삭제된 파일은 복구할 수 없으니, 실행 전 반드시 데이터를 백업하세요!")
    print("="*50)
    
    # 사용자 확인 절차
    confirm = input(f"\n경고: '{BASE_TRAIN_DIR}' 경로와 그 하위 폴더에 대해 작업을 진행합니다. \n"
                    f"'{', '.join(sorted(list(classes_to_keep))[:5])}...' 등 총 {len(classes_to_keep)}개 폴더의 이미지는 보존되고, \n"
                    f"나머지 폴더의 이미지들은 삭제됩니다. \n"
                    "정말로 계속하시겠습니까? (yes 입력 시 진행): ")

    if confirm.lower() == 'yes':
        print("\n이미지 삭제 작업을 시작합니다...")
        delete_images_from_other_folders(BASE_TRAIN_DIR, classes_to_keep)
    else:
        print("\n작업이 사용자에 의해 취소되었습니다.")