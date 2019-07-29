from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load the trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    print(img_path)
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (152, 632))
    img = img.astype(np.float32) / 255.0
    img = img[None]
    img = img[...,None]
    preds = model.predict(img)[0][...,0]
    preds = (255*preds).astype(np.uint8)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Output file path
        file_output_path = os.path.join(basepath, 'uploads', 'output.png')
        # Make prediction
        preds = model_predict(file_path, model)

        # Write to uploads directory
        cv2.imwrite(file_output_path, preds)

        # Delete uploaded file
        os.remove(file_path)
        return file_output_path
    return None


# Callback to grab an image given a local path
@app.route('/get_image')
def get_image():
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return send_file(path, mimetype='image/' + ext[1:])


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 80), app)
    http_server.serve_forever()
