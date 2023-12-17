# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:53:52 2020

@author: dreve
"""

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

import numpy as np
import cv2
import matplotlib.pyplot as plt


width_shape = 280
height_shape = 280


names = ['Apple__Apple_scab',' Apple__Black_rot', 'Apple__Cedar_apple_rust','Apple__healthy', 'Blueberry__healthy', 'Cherry_(including_sour)__healthy', 'Cherry_(including_sour)__Powdery_mildew', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust_', 'Corn_(maize)__healthy', 'Corn_(maize)__Northern_Leaf_Blight', 'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__healthy', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy', 'Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight', 'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew', 'Strawberry__healthy', 'Strawberry__Leaf_scorch', 'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__healthy', 'Tomato__Late_blight', 'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']


modelt = load_model('C:/Users/karlo/OneDrive/Escritorio/API_DeepLearning-master/models/model_VGG16.h5')
print("Modelo cargado exitosamente")

imaget_path = "apple_healty.jpg"
imaget=cv2.resize(cv2.imread(imaget_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)

xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)

print("Predicción")
preds = modelt.predict(xt)

print("Predicción:", names[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(imaget),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()