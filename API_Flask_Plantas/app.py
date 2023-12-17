from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from werkzeug.utils import secure_filename
import os

width_shape = 280
height_shape = 280

names = ['Apple__Apple_scab',' Apple__Black_rot', 'Apple__Cedar_apple_rust','Apple__healthy', 'Blueberry__healthy', 'Cherry_(including_sour)__healthy', 'Cherry_(including_sour)__Powdery_mildew', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust_', 'Corn_(maize)__healthy', 'Corn_(maize)__Northern_Leaf_Blight', 'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__healthy', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy', 'Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight', 'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew', 'Strawberry__healthy', 'Strawberry__Leaf_scorch', 'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__healthy', 'Tomato__Late_blight', 'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']

# Definimos una instancia de Flask
app = Flask(__name__)

# Ruta del modelo preentrenado
MODEL_PATH = 'C:/Users/karlo/OneDrive/Escritorio/API_DeepLearning-master/models/model_VGG16.h5'

# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)

print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Realizamos la predicción usando la imagen cargada y el modelo
def model_predict(img_path, model):
    img = cv2.resize(cv2.imread(img_path), (width_shape, height_shape), interpolation=cv2.INTER_AREA)
    x = np.asarray(img)
    x = preprocess_input(x)
    #x = x.reshape((1, -1))
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo del request
        f = request.files['file']

        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])

        # Enviamos el resultado de la predicción
        result = str(names[np.argmax(preds)])
        return result

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
