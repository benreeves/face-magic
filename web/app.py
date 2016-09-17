from flask import Flask, render_template, redirect, flash, url_for, request
from werkzeug.utils import secure_filename
import dlib
import time, atexit
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import os, random, shutil
import numpy as np
import evaluate
import keras_model
import feature_detector

"""Configuration Region"""
WEBROOT = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = WEBROOT + '/static/img/tmp'
ALLOWED_EXTENSIONS=(['jpeg', 'png', 'jpg'])
RESOURCE_FOLDER = WEBROOT + '/res'
app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['WEBROOT']=WEBROOT
app.config['RESOURCE_FOLDER']=RESOURCE_FOLDER

"""Initialize some application-level pieces"""
predictor = keras_model.Predictor()
predictor.load_weights()
featureDetector = feature_detector.FeatureDetector(app.config['UPLOAD_FOLDER'], app.config['RESOURCE_FOLDER'] + '/shape_predictor_68_face_landmarks.dat') 
def purge_temp_files():
    for tmp_file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, tmp_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path);
        except Exception as e:
            print(e)
            #TODO enable logging.
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=purge_temp_files,
    trigger=IntervalTrigger(hours=6),
    id='purging_temp',
    name='Purges temporarily stored images',
    replace_existing=True)
# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

def evaluate_face(imgPath = '', remote=True):
    results = []
    if not imgPath:
        res= 'You did not upload an image'
    elif not os.path.isfile(imgPath):
        res = 'You did not upload a valid image'
    else:
        faces = evaluate.image_to_numpy(imgPath)
        if type(faces) is str:
            res = faces #this is a message string in this case
        else:
            for face in faces:
                results.append(predictor.get_gender(face))
                res = 'Our best guess is that the person is: ' + results[0] #TODO need to handle multiple faces

    if remote == True:
        return render_template('_result.html', message=res)
    else:
        if type(faces) is not str:
            fig_path = featureDetector.save_landmarks(imgPath)
        else:
            fig_path = ''
        return render_template('result.html', message=res, fig=fig_path)
    
def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS #TODO create a regex to parse this better

def populate_samples():
    return random.sample(os.listdir(WEBROOT+ '/static/img/samples'), 4)


@app.route('/')
def main():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('no file')
            return redirect(request_url)
        file = request.files['file']
        if file.filename == '':
            flash('no file provided')
            redirect(request_url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #Check if we've already saved the image
            if os.path.isfile(file_path):
                return evaluate_face(file_path, False)
            file.save(os.path.join(file_path))
            result = evaluate_face(file_path, False)
            return result

    if request.method == 'GET':
        if 'img' not in request.args:
            return render_template('upload.html', samples=populate_samples())
        imgPath = request.args.get('img', '')
        if not imgPath:
            flash('something went wrong')
            redirect(request_url)
        file_path = os.path.join(app.config['WEBROOT'], imgPath)
        return evaluate_face(file_path)

    return render_template('upload.html', samples=populate_samples())

@app.route('/eval')
def eval():
    """Evaluates the age and gender of the provided face"""
    rtn = evaluate_face()
    return redirect(url_for('upload'))

