import os
import shutil

def move_images(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        print(f"❌ 소스 폴더 없음: {src_dir}")
        return
    if not os.path.exists(dst_dir):
        print(f"📁 대상 폴더 없음, 생성함: {dst_dir}")
        os.makedirs(dst_dir)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    moved_count = 0

    for fname in os.listdir(src_dir):
        if os.path.splitext(fname)[1].lower() in image_exts:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            base, ext = os.path.splitext(fname)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)
            moved_count += 1

    print(f"✅ {moved_count}개의 파일을 '{src_dir}' → '{dst_dir}'로 이동 완료.")

if __name__ == "__main__":
    base_path = "/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/train_final_test"

    class_pairs = [
        ("K5_하이브리드_3세대_2020_2023", "K5_3세대_하이브리드_2020_2022"),
        ("디_올_뉴_니로_2022_2025", "디_올뉴니로_2022_2025"),
        ("박스터_718_2017_2024", "718_박스터_2017_2024"),
        ("RAV4_2016_2018", "라브4_4세대_2013_2018"),
        ("RAV4_5세대_2019_2024", "라브4_5세대_2019_2024")
    ]

    for src_name, dst_name in class_pairs:
        src_folder = os.path.join(base_path, src_name)
        dst_folder = os.path.join(base_path, dst_name)
        move_images(src_folder, dst_folder)
