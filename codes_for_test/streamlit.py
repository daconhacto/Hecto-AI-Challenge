import streamlit as st
import json
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 파일 경로
json_path = '/project/ahnailab/jys0207/CP/tjrgus5/hecto/work_directories/convnext_mosaic_test_lr1e-5_retrain/wrong_examples/Fold_1_Epoch_5_wrong_examples.json'

# JSON 파일 로딩
with open(json_path, 'r', encoding='utf-8') as f:
    wrong_images_dict = json.load(f)

# 타이틀
st.title("카테고리별 틀린 이미지 슬라이드")

# 카테고리 목록
categories = list(wrong_images_dict.keys())

# 세션 상태 초기화
if 'category_index' not in st.session_state:
    st.session_state.category_index = 0
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

# 카테고리 선택 박스
selected_category = st.selectbox("카테고리를 선택하세요:", categories, index=st.session_state.category_index)

# 카테고리 변경 시 인덱스 초기화
if selected_category != categories[st.session_state.category_index]:
    st.session_state.category_index = categories.index(selected_category)
    st.session_state.image_index = 0

# 카테고리 변경 버튼
col_cat1, col_cat2 = st.columns(2)
with col_cat1:
    if st.button("⬅ 이전 카테고리"):
        if st.session_state.category_index > 0:
            st.session_state.category_index -= 1
            st.session_state.image_index = 0
with col_cat2:
    if st.button("다음 카테고리 ➡"):
        if st.session_state.category_index < len(categories) - 1:
            st.session_state.category_index += 1
            st.session_state.image_index = 0

# 현재 선택된 카테고리 데이터
current_category = categories[st.session_state.category_index]
image_info_list = wrong_images_dict[current_category]
total_images = len(image_info_list)

# 이미지 넘기기 버튼
col_img1, col_img2 = st.columns(2)
with col_img1:
    if st.button("◀ 이전 이미지"):
        st.session_state.image_index = max(st.session_state.image_index - 1, 0)
with col_img2:
    if st.button("다음 이미지 ▶"):
        st.session_state.image_index = min(st.session_state.image_index + 1, total_images - 1)

# 현재 이미지 정보
img_info = image_info_list[st.session_state.image_index]
img_path = img_info.get('image_path')

answer_key = 'model_answer' if 'model_answer' in img_info else 'correct_answer'
model_answer = img_info.get(answer_key, 'N/A')

# 이미지 표시
st.markdown(f"**카테고리: {current_category}**")
st.markdown(f"**이미지 {st.session_state.image_index + 1} / {total_images}**")
st.markdown(f"**모델이 예측한 클래스: {model_answer}**")

if os.path.exists(img_path):
    st.image(Image.open(img_path), caption=os.path.basename(img_path))
else:
    st.warning(f"이미지를 찾을 수 없습니다: {img_path}")