import os
from flask import Flask, request, render_template
from flask_cors import CORS
import numpy as np
import cv2 as cv
from tensorflow.keras import models

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = models.load_model('image_classifier.model')
class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))

    prediction = model.predict(np.array([img]) / 255.0)
    index = np.argmax(prediction)
    prediction_text = f'Prediction is: {class_names[index]}'

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
