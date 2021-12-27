import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import time

idx2label = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute"
}

model = tf.keras.models.load_model("best.hdf5")

def predict(image):
    image = tf.image.resize(image, (160, 160))
    batch = tf.expand_dims(image, 0)
    res = model(batch)
    my_bar = st.progress(0)
    for precent in range(100):
        time.sleep(0.01)
        my_bar.progress(precent + 1)
    return idx2label[tf.argmax(res[0]).numpy()], tf.sigmoid(res[0])[tf.argmax(res[0]).numpy()].numpy()


st.markdown("<html><h1><center>Imagenette classifier</center></h1></html>",
            unsafe_allow_html=True)

labels = list(idx2label.values())


res = st.sidebar.selectbox("Image from?", options=["URL", "Local"])

if res == "Local":

    st.markdown("<html><h2><center>Upload an image</center></h2></html>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(label='', type=["jpg", "png"])

    bt = st.button("Predict")

    if(uploaded_file is not None):
        st.image(uploaded_file, use_column_width=True)
        pred, confidence = "", 0
        if bt:
            with st.spinner("Classifying..."):
                img = Image.open(uploaded_file)
                img = np.array(img)

                res = tf.convert_to_tensor(img, dtype=tf.float32)

                with tf.device("/CPU:0"):
                    conf_idx = tf.argmax(tf.sigmoid(res[0]))
                    pred, confidence = predict(res)

            st.success(
                f"It's a {pred} with a confidence of {confidence * 100:.4f}%")

else:
    # get image from URL
    st.markdown("<html><h2><center>Enter an image URL</center></h2></html>",
                unsafe_allow_html=True)

    url = st.text_input("Image URL")
    bt = st.button("Predict")

    if url:
        st.image(url, use_column_width=True)
        pred, confidence = "", 0
        if bt:
            with st.spinner("Classifying..."):
                img = requests.get(url).content
                img = Image.open(BytesIO(img))
                img = np.array(img)

                res = tf.convert_to_tensor(img, dtype=tf.float32)

                with tf.device("/CPU:0"):
                    conf_idx = tf.argmax(tf.sigmoid(res[0]))
                    pred, confidence = predict(res)

            st.success(
                f"It's a {pred} with a confidence of {confidence * 100:.4f}%")

st.sidebar.markdown(f"<p>Currently classification supported for the follwing classes below</p>",
                    unsafe_allow_html=True)

# Create a unordered list from the labels
st.sidebar.markdown(f"<ul>{''.join(['<li>{}</li>'.format(label) for label in labels])}</ul>",
                    unsafe_allow_html=True)
