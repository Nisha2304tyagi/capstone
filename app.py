from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from your_script import brand_insights  # Import your analysis script

# Initialize the Flask app
app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        return "No file part", 400

    file = request.files['csv_file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Retrieve additional inputs
        analysis_type = request.form.get('analysis_type')
        top_n = int(request.form.get('top_n', 10))
        filter_negative = 'filter_negative' in request.form

        # Process the file
        df = pd.read_csv(file_path)
        brand_insights(df, top_n=top_n, filter_negative=filter_negative)

        return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
