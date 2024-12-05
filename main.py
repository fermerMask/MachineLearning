import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title("手書き文字認識")
st.sidebar.header("モデルの選択")
model_dir = st.sidebar.text_input("モデルフォルダのパスを入力","./model")
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    selected_model = st.sidebar.selectbox("使用するモデルを選択",model_files)
    if selected_model:
        model_path = os.path.join(model_dir, selected_model)
        model = torch.load(model_path,map_location=torch.device("cpu"))
        st.sidebar.success(f"モデル{selected_model}が読み込まれました．")
    else:
        st.sidebar.error("モデルファイルが見つかりません．正しいパスを入力してください")

st.sidebar.header("キャンバス設定")
stroke_width = st.sidebar.slider("線の太さ:",1,25,5)
stroke_color = st.color_picker("線の色", "#000000")
bg_color = st.sidebar.color_picker("背景の色", "#FFFFFF")

st.write("## 手書き文字を書いてください")
canvas_result = st_canvas(
    fill_color=bg_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    height=200 ,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"デバイス:{'GPU' if device.type == 'cuda' else 'CPU'}")
model = model.to(device)

if canvas_result.image_data is not None: 
    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype("uint8"))
    image = ImageOps.grayscale(image)  
    image = image.resize((28, 28)) 
    resized_image = np.array(image)
    input_tensor = torch.tensor(resized_image, dtype=torch.float32)

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    input_tensor = input_tensor.view(input_tensor.size(0), -1).to(device)  # [1, 784]に変換

    if st.button("認識開始") and selected_model:
        with torch.no_grad():
            # 推論
            output = model(input_tensor)  # モデルに入力
            predict_label = torch.argmax(output, axis=1).item()  # 最も確率の高いラベルを取得
        st.write(f"認識結果: {predict_label}")