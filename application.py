import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import urllib.request
from flask import flash, request, redirect, url_for, render_template
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from flask_wtf import FlaskForm

application = Flask(__name__)
app = application

application.secret_key = "secret key"
UPLOAD_FOLDER = 'Static/uploads/'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

n_model = load_model('model.h5')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

    
@application.route('/', methods=['GET',"POST"])
@application.route('/home', methods=['GET',"POST"])
def home():
    #if file and allowed_file(file.filename):
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        img_filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        #file.save(img_filename)
        print(filepath)
        img = cv2.imread(filepath)
        img_data = np.array(img)
        print(img_data.shape)
        if img_data.shape != (50,50,3):
        
            img_resize = cv2.resize(img_data, (50, 50))
            img_data_ex = np.expand_dims(img_resize, axis = 0)
        #print(img)
        #print(img_resize.shape)
        
        else:
            img_data_ex = np.expand_dims(img_data, axis = 0)
        #img_data_ex = np.expand_dims(img_resize, axis = 0)
        
        print(img_data_ex.shape)
        #img = read_image(filepath) 
        class_prediction=n_model.predict(img_data_ex) 
        classes_x=np.argmax(class_prediction,axis=1)
        pred_dat=classes_x[0]
        print("Index of the maximum value: ", classes_x)
        Dict_values={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '_'}
        if pred_dat in Dict_values:
            pred_val= Dict_values[pred_dat]
            return render_template("index.html",form = form,result=pred_val)

        #return render_template("index.html",form = form,result=classes_x)

    return render_template('index.html', form=form)

if __name__ == "__main__":
    application.run(debug=True)