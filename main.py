from flask import Flask,render_template, flash, request, redirect, url_for,send_from_directory
from pyspark import SparkContext
from pyspark.sql import SparkSession
from werkzeug.utils import secure_filename
import os

from controller.Prediction import prediction

#sc = SparkContext('local[*]', 'POC')

#spark = SparkSession.builder.appName("POC").master("local[*]").getOrCreate()

UPLOAD_FOLDER = 'resource/'
ALLOWED_EXTENSIONS = {'csv'}



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route('/')
def hello():
    prediction.main(None)
    print("Hello World!")
    return "Hello World!"

@app.route("/home")
def home():
    return render_template("home.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return render_template("home.html")

@app.route('/uploads/<name>')
def download_file(name):
    prediction.main(None,os.path.join(app.config['UPLOAD_FOLDER'], name))
    return send_from_directory(app.config["UPLOAD_FOLDER"], 'AnomalyPredictions.csv')


