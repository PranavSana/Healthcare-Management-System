from flask import Flask, render_template, request
import os

# --------------------
# IMPORT ML UTILITIES
# --------------------
from utils.predict import predict_heart_disease
from utils.lung_predict import predict_lung_cancer
from utils.brain_predict import predict_brain_tumor

app = Flask(__name__)

# --------------------
# UPLOAD FOLDER
# --------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------
# HOME PAGE
# --------------------
@app.route('/')
def index():
    return render_template('index.html')


# --------------------
# DASHBOARD
# --------------------
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# --------------------
# HEART DISEASE
# --------------------
@app.route('/disease1', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        age        = request.form.get('age')
        bp         = request.form.get('blood_pressure')
        cholesterol = request.form.get('cholesterol')
        heart_rate = request.form.get('heart_rate')
        patient_name = request.form.get('patient_name', '').strip()

        result = predict_heart_disease(age, bp, cholesterol, heart_rate)

        return render_template('result.html', result=result, patient_name=patient_name)

    return render_template('disease1.html')


# --------------------
# LUNG CANCER
# --------------------
@app.route('/disease2', methods=['GET', 'POST'])
def lung_cancer():
    if request.method == 'POST':
        file = request.files.get('ct_image')
        patient_name = request.form.get('patient_name', '').strip()

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            result = predict_lung_cancer(file_path)
            return render_template('result.html', result=result, patient_name=patient_name)

    return render_template('disease2.html')


# --------------------
# BRAIN TUMOR
# --------------------
@app.route('/disease3', methods=['GET', 'POST'])
def brain_tumor():
    if request.method == 'POST':
        file = request.files.get('mri_image')
        patient_name = request.form.get('patient_name', '').strip()

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            result = predict_brain_tumor(file_path)
            return render_template('result.html', result=result, patient_name=patient_name)

    return render_template('disease3.html')


# --------------------
# RUN APPLICATION
# --------------------
if __name__ == '__main__':
    app.run(debug=True)