import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

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

st.title("Imagenette classifier")

st.write(idx2label)

image = st.file_uploader("Upload an image!", type=['jpg', 'png'])

model = tf.keras.models.load_model("best.hdf5")


def predict(image):
    image = tf.image.resize(image, (160, 160))
    batch = tf.expand_dims(image, 0)
    res = model(batch)
    return idx2label[tf.argmax(res[0]).numpy()], tf.sigmoid(res[0])[tf.argmax(res[0]).numpy()].numpy()


if(image is not None):
    pred, confidence = "", 0
    with st.spinner("Classifying..."):
        img = Image.open(image)
        img = np.array(img)

        res = tf.convert_to_tensor(img, dtype=tf.float32)

        with tf.device("/CPU:0"):
            conf_idx = tf.argmax(tf.sigmoid(res[0]))
            pred, confidence = predict(res)

    st.success(f"It's a {pred} with a confidence of {confidence * 100:.4f}%")
    st.image(img, use_column_width=True)
