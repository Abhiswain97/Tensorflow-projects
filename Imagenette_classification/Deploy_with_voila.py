#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image
from ipywidgets import widgets
import io


# In[2]:


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


# In[3]:


def classify(image, model):
    image = tf.convert_to_tensor(np.array(image)).numpy()
    image = tf.image.resize(image, (160, 160))
    batch = tf.expand_dims(image, 0)
    res = model(batch)
    conf_idx = tf.argmax(tf.sigmoid(res[0]))
    print(f"It's a: {idx2label[tf.argmax(res[0]).numpy()]} with a confidence of {tf.sigmoid(res[0])[conf_idx] * 100:.3f}%")


# In[4]:


from ipywidgets import FileUpload
upload = FileUpload()
upload


# In[5]:


model = tf.keras.models.load_model("best.hdf5")


# In[6]:


button = widgets.Button(description='Classify!')
out = widgets.Output()

def on_button_clicked(_):
    with out:
        with tf.device('/CPU:0'):
            data = upload.data
            image = Image.open(io.BytesIO(data[-1]))
            classify(image, model)
        
button.on_click(on_button_clicked)
widgets.VBox([button,out])


# In[ ]:




