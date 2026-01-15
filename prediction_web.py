import streamlit as st 
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import imageio
import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from imgaug import augmenters as iaa
import json

# 載入'類別代碼'對應'鈔票種類'的字典
# with open('label_dict.json', 'r') as f:
with open('label_dict_cnn.json', 'r') as f:
    label_dict = json.load(f)
print(label_dict)

# 載入模型
# model = tf.keras.models.load_model('cash.keras', compile=False)
model = tf.keras.models.load_model('cash_cnn.keras', compile=False)

st.title("上傳紙鈔圖片辨識")

uploaded_file = st.file_uploader("上傳圖片(.jpg, .png)", type=["jpg","png"])
img_size = (300, 600, 3)
if uploaded_file is not None:
    # image = imageio.v3.imread(uploaded_file)
    # resize2 = iaa.Resize({"height": 500, "width": 1024})
    # image = resize2(image=image)
    # image = image / 255.0
    # X1 = image.reshape((-1, *image.shape))
    # predictions = np.argmax(model.predict(X1))
    # st.markdown(f"# {label_dict[predictions]}")
    # st.image(image)

    
    image = io.imread(uploaded_file)
    image = resize(image, img_size[:-1])    
    X1 = image.reshape(1,*img_size) # / 255
    st.write("predict...")
    predictions = np.argmax(model.predict(X1))
    st.markdown(f"# {label_dict[str(predictions)]}")
    st.image(image)
