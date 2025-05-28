import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob

# --- 사용자 입력 ---
root = '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/test'
csv_files = ['/project/ahnailab/jys0207/CP/tjrgus5/hecto/submissions/weight_emsemble_0.09896.csv',
             '/project/ahnailab/jys0207/CP/tjrgus5/hecto/submissions/submission_eva02_large_patch14_448.mim_m38m_ft_in1k_fold1_mixup_mosaic_cutmix.csv']
csv_file_names = [
    'best_ensemble',
    'eva_0.1258'
]
image_ext = '.jpg'


dfs = []
for path, name in zip(csv_files, csv_file_names):
    df = pd.read_csv(path)
    dfs.append((name, df))

# --- 공통 ID 목록 및 disagree_ids 추출 (최초 1회만) ---
if "disagree_ids" not in st.session_state:
    # 공통 ID 추출
    common_ids = list(set(dfs[0][1]['ID']).intersection(set(dfs[1][1]['ID'])))
    common_ids.sort()

    # top-1이 서로 다른 경우만 저장
    disagree_ids = []
    for img_id in common_ids:
        row1 = dfs[0][1][dfs[0][1]['ID'] == img_id].iloc[0]
        row2 = dfs[1][1][dfs[1][1]['ID'] == img_id].iloc[0]

        probs1 = row1.drop(labels=['ID']).astype(float)
        probs2 = row2.drop(labels=['ID']).astype(float)

        top1_class_1 = probs1.idxmax()
        top1_class_2 = probs2.idxmax()

        if top1_class_1 != top1_class_2:
            disagree_ids.append(img_id)

    st.session_state.disagree_ids = disagree_ids
    st.session_state.id_index = 0  # 초기 인덱스 설정

# 이후에는 session_state에서 바로 사용
disagree_ids = st.session_state.disagree_ids

# --- session_state 사용한 탐색 기능 ---
if 'id_index' not in st.session_state:
    st.session_state.id_index = 0

if len(disagree_ids) == 0:
    st.info("두 모델의 top-1 예측이 모두 동일합니다.")
    st.stop()

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ 이전"):
        st.session_state.id_index = max(0, st.session_state.id_index - 1)

with col3:
    if st.button("다음 ➡️"):
        st.session_state.id_index = min(len(disagree_ids) - 1, st.session_state.id_index + 1)

with col2:
    selected_id = st.selectbox("확인할 이미지 ID (Top1 불일치)", disagree_ids, index=st.session_state.id_index)
    st.session_state.id_index = disagree_ids.index(selected_id)

# --- 이미지 출력 ---
image_path = os.path.join(root, selected_id + image_ext)
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption=selected_id, use_column_width=True)
else:
    st.error(f"이미지를 찾을 수 없습니다: {image_path}")
    st.stop()

# --- 모델별 Top-3 확률 표시 ---
cols = st.columns(2)

for idx, (model_name, df) in enumerate(dfs):
    row = df[df['ID'] == selected_id].iloc[0]
    probs = row.drop('ID').astype(float)
    top3 = probs.sort_values(ascending=False).head(3)

    with cols[idx]:
        st.subheader(model_name)
        for cls, prob in top3.items():
            st.write(f"**{cls}**: {prob*100:.2f}%")
