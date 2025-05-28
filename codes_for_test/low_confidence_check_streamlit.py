import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob

# --- 사용자 입력 ---
root = '/project/ahnailab/jys0207/CP/lexxsh_project_3/hecto/test'
csv_file = '/project/ahnailab/jys0207/CP/tjrgus5/hecto/submissions/weight_emsemble_0.09896.csv'
csv_file_name = 'best_ensemble'
image_ext = '.jpg'
max_threshold = 0.7 # 70%보다 자신감 없게 예측한 케이스 보기
min_threshold = 0.3


df = pd.read_csv(csv_file)

# --- 공통 ID 목록 및 disagree_ids 추출 (최초 1회만) ---
if "low_ids" not in st.session_state:
    # 공통 ID 추출
    ids = list(set(df['ID']))
    ids.sort()

    # top-1이 서로 다른 경우만 저장
    low_ids = []
    for img_id in ids:
        row1 = df[df['ID'] == img_id].iloc[0]

        probs = row1.drop(labels=['ID']).astype(float)

        top1_class_1 = probs.idxmax()

        if probs[top1_class_1] < max_threshold and probs[top1_class_1] > min_threshold:
            low_ids.append(img_id)

    st.session_state.low_ids = low_ids
    st.session_state.id_index = 0  # 초기 인덱스 설정

# 이후에는 session_state에서 바로 사용
low_ids = st.session_state.low_ids

# --- session_state 사용한 탐색 기능 ---
if 'id_index' not in st.session_state:
    st.session_state.id_index = 0

if len(low_ids) == 0:
    st.info(f"모든 예측에 대한 confidence가 {max_threshold}를 넘습니다.")
    st.stop()

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ 이전"):
        st.session_state.id_index = max(0, st.session_state.id_index - 1)

with col3:
    if st.button("다음 ➡️"):
        st.session_state.id_index = min(len(low_ids) - 1, st.session_state.id_index + 1)

with col2:
    selected_id = st.selectbox(f"확인할 이미지 ID (Top1 {max_threshold}미만)", low_ids, index=st.session_state.id_index)
    st.session_state.id_index = low_ids.index(selected_id)

# --- 이미지 출력 ---
image_path = os.path.join(root, selected_id + image_ext)
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption=selected_id, use_column_width=True)
else:
    st.error(f"이미지를 찾을 수 없습니다: {image_path}")
    st.stop()

# --- Top-3 확률 표시 ---
row = df[df['ID'] == selected_id].iloc[0]
probs = row.drop('ID').astype(float)
top3 = probs.sort_values(ascending=False).head(3)


st.subheader(csv_file_name)
for cls, prob in top3.items():
    st.write(f"**{cls}**: {prob*100:.2f}%")
