import os
import glob

# 파일 리스트를 딕셔너리로 변환
file_list = [
    "2시리즈_그란쿠페_F44_2020_2024_0042.jpg",
    "2시리즈_액티브_투어러_U06_2022_2024_0004.jpg",
    "3시리즈_F30_2013_2018_0069.jpg",
    "3시리즈_F30_2013_2018_0036.jpg",
    "4시리즈_F32_2014_2020_0027.jpg",
    "4시리즈_G22_2024_2025_0031.jpg",
    "5시리즈_G60_2024_2025_0010.jpg",
    "6시리즈_GT_G32_2018_2020_0018.jpg",
    "7시리즈_F01_2009_2015_0029.jpg",
    "7시리즈_F01_2009_2015_0044.jpg",
    "7시리즈_G11_2016_2018_0040.jpg",
    "911_992_2020_2024_0006.jpg",
    "911_992_2020_2024_0030.jpg",
    "5008_2세대_2021_2024_0055.jpg",
    "5008_2세대_2021_2024_0051.jpg",
    "A_클래스_W177_2020_2025_0034.jpg",
    "A8_D5_2018_2023_0084.jpg",
    "CLS_클래스_C257_2019_2023_0021.jpg",
    "E_클래스_W212_2010_2016_0069.jpg",
    "ES300h_7세대_2019_2026_0028.jpg",
    "F150_2004_2021_0046.jpg",
    "G_클래스_W463_2009_2017_0011.jpg",
    "G_클래스_W463b_2019_2025_0030.jpg",
    "G_클래스_W463b_2019_2025_0049.jpg",
    "GLB_클래스_X247_2020_2023_0008.jpg",
    "GLE_클래스_W167_2019_2024_0068.jpg",
    "GLS_클래스_X167_2020_2024_0013.jpg",
    "K3_2013_2015_0045.jpg",
    "K5_2세대_2016_2018_0007.jpg",
    "K5_3세대_2020_2023_0081.jpg",
    "Q5_FY_2021_2024_0032.jpg",
    "Q7_4M_2020_2023_0011.jpg",
    "Q30_2017_2019_0074.jpg",
    "Q30_2017_2019_0075.jpg",
    "Q50_2014_2017_0031.jpg",
    "RAV4_5세대_2019_2024_0020.jpg",
    "S_클래스_W223_2021_2025_0008.jpg",
    "S_클래스_W223_2021_2025_0071.jpg",
    "SM7_뉴아트_2008_2011_0053.jpg",
    "X4_F26_2015_2018_0068.jpg",
    "X7_G07_2019_2022_0052.jpg",
    "XF_X260_2016_2020_0023.jpg",
    "그랜드_체로키_WL_2021_2023_0018.jpg",
    "글래디에이터_JT_2020_2023_0075.jpg",
    "뉴_A6_2012_2014_0046.jpg",
    "뉴_CC_2012_2016_0002.jpg",
    "뉴_ES300h_2013_2015_0000.jpg",
    "뉴_카이엔_2011_2018_0065.jpg",
    "더_기아_레이_EV_2024_2025_0078.jpg",
    "더_뉴_K3_2세대_2022_2024_0001.jpg",
    "더_뉴_QM6_2024_2025_0040.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0078.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0079.jpg",
    "더_뉴_그랜드_스타렉스_2018_2021_0080.jpg",
    "더_뉴_스파크_2019_2022_0040.jpg",
    "더_뉴_아반떼_2014_2016_0031.jpg",
    "더_뉴_코나_2021_2023_0081.jpg",
    "더_뉴_파사트_2012_2019_0067.jpg",
    "더_올뉴투싼_하이브리드_2021_2023_0042.jpg",
    "디_올뉴그랜저_2023_2025_0039.jpg",
    "라브4_4세대_2013_2018_0014.jpg",
    "라브4_4세대_2013_2018_0022.jpg",
    "레니게이드_2019_2023_0041.jpg",
    "레이_2012_2017_0063.jpg",
    "레인지로버_4세대_2018_2022_0048.jpg",
    "레인지로버_5세대_2023_2024_0030.jpg",
    "레인지로버_스포츠_2세대_2018_2022_0014.jpg",
    "레인지로버_스포츠_2세대_2018_2022_0017.jpg",
    "리얼_뉴_콜로라도_2021_2022_0067.jpg",
    "마칸_2019_2021_0035.jpg",
    "머스탱_2015_2023_0086.jpg",
    "박스터_718_2017_2024_0011.jpg",
    "베뉴_2020_2024_0005.jpg",
    "싼타페_TM_2019_2020_0009.jpg",
    "아반떼_MD_2011_2014_0009.jpg",
    "아반떼_MD_2011_2014_0082.jpg",
    "아반떼_MD_2011_2014_0081.jpg",
    "아반떼_N_2022_2023_0035.jpg",
    "아반떼_N_2022_2023_0064.jpg",
    "아베오_2012_2016_0052.jpg",
    "익스플로러_2016_2017_0072.jpg",
    "카이엔_PO536_2019_2023_0035.jpg",
    "카이엔_PO536_2019_2023_0054.jpg",
    "컨티넨탈_GT_3세대_2018_2023_0007.jpg",
    "콰트로포르테_2017_2022_0074.jpg",
    "타이칸_2021_2025_0003.jpg",
    "타이칸_2021_2025_0065.jpg",
    "파나메라_2010_2016_0000.jpg",
    "파나메라_2010_2016_0036.jpg",
    "프리우스_4세대_2019_2022_0052.jpg",
]

# 파일명에서 폴더명과 번호 추출하여 딕셔너리 생성
def parse_file_list(file_list):
    files_to_delete = {}
    
    for filename in file_list:
        # 파일명에서 .jpg 제거
        name_without_ext = filename.replace('.jpg', '')
        
        # 마지막 '_' 기준으로 분리하여 번호 추출
        parts = name_without_ext.split('_')
        number = int(parts[-1])  # 마지막 부분이 번호
        folder_name = '_'.join(parts[:-1])  # 나머지가 폴더명
        
        if folder_name not in files_to_delete:
            files_to_delete[folder_name] = []
        files_to_delete[folder_name].append(number)
    
    return files_to_delete

# 삭제할 파일 목록 설정
files_to_delete = parse_file_list(file_list)

# 기본 경로 설정 (실제 경로로 수정해야 함)
base_dir = "/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/train_final_test"

print("삭제할 파일 목록:")
print("=" * 50)
for folder, numbers in files_to_delete.items():
    print(f"{folder}: {numbers}")
print("=" * 50)

# 실제 삭제 여부 확인
delete_mode = True  # True로 변경하면 실제 삭제됩니다

deleted_count = 0
not_found_count = 0

# 각 폴더에서 해당 번호의 파일 삭제
for folder, numbers in files_to_delete.items():
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.exists(folder_path):
        print(f"\n📁 폴더: {folder}")
        for number in numbers:
            # 파일명 패턴 생성 (예: GLE_클래스_W167_2019_2024_0068.jpg)
            file_pattern = f"{folder}_{number:04d}.jpg"
            file_path = os.path.join(folder_path, file_pattern)
            
            # 정확한 파일명으로 먼저 찾기
            if os.path.exists(file_path):
                if delete_mode:
                    os.remove(file_path)
                    print(f"  ✅ 삭제 완료: {file_pattern}")
                else:
                    print(f"  🔍 삭제 예정: {file_pattern}")
                deleted_count += 1
            else:
                # glob을 사용하여 패턴에 맞는 파일 찾기 (대소문자 구분 없이)
                matching_files = glob.glob(os.path.join(folder_path, f"{folder}_*{number:04d}*.jpg"))
                
                if matching_files:
                    for file in matching_files:
                        filename = os.path.basename(file)
                        if delete_mode:
                            os.remove(file)
                            print(f"  ✅ 삭제 완료: {filename}")
                        else:
                            print(f"  🔍 삭제 예정: {filename}")
                        deleted_count += 1
                else:
                    print(f"  ❌ 파일을 찾을 수 없음: {file_pattern}")
                    not_found_count += 1
    else:
        print(f"❌ 폴더를 찾을 수 없음: {folder_path}")
        not_found_count += len(numbers)

print(f"\n📊 결과 요약:")
print(f"{'삭제 완료' if delete_mode else '삭제 예정'}: {deleted_count}개")
print(f"찾을 수 없음: {not_found_count}개")
print(f"전체: {len(file_list)}개")

if not delete_mode:
    print(f"\n⚠️  현재는 미리보기 모드입니다.")
    print(f"실제로 삭제하려면 'delete_mode = True'로 변경하세요.")
else:
    print(f"\n✅ 삭제 작업이 완료되었습니다.")