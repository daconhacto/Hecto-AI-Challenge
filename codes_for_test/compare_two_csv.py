import pandas as pd


p1 = "/project/ahnailab/jys0207/CP/tjrgus5/hecto/submission_convnext_base.fb_in22k_ft_in1k_384_tta.csv"
p2 = "/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/emsemble.csv"

def compute_max_feature_match_rate(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # ID 기준 정렬 (동일한 순서 가정 or 정렬 필요)
    df1 = df1.sort_values('ID').reset_index(drop=True)
    df2 = df2.sort_values('ID').reset_index(drop=True)

    assert all(df1['ID'] == df2['ID']), "두 CSV 파일의 ID가 다릅니다."

    # Feature 컬럼만 추출
    feature_cols = [col for col in df1.columns if col != 'ID']
    feat1 = df1[feature_cols].values
    feat2 = df2[feature_cols].values

    # 각 row마다 가장 큰 feature 인덱스 추출
    max_idx_1 = feat1.argmax(axis=1)
    max_idx_2 = feat2.argmax(axis=1)

    # 일치율 계산
    match = (max_idx_1 == max_idx_2)
    match_rate = match.mean()

    print(f"✅ 총 샘플 수: {len(match)}")
    print(f"🎯 최댓값 feature 일치 개수: {match.sum()}")
    print(f"📊 최댓값 feature 일치율: {match_rate:.4f}")

    return match_rate

print(compute_max_feature_match_rate(p1, p2))