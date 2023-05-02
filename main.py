import numpy as np
from flask import Flask, request, jsonify
import pickle

model = pickle.load(open('pickle_sem8.pickle', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Heart Disease Prediction App"


@app.route('/predict', methods=['POST'])
# 'cp','thalach','slope','restecg','chol','trestbps','fbs','oldpeak'
# [0,108,1,0,250,160,1,1.5] 0
# [3,150,0,0,233,145,1,2.3] 1

def predict():
    age = request.form.get("age")
    gender = request.form.get('gender')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exe = request.form.get('exe')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')


     # result = {'cp':cp,'thalach':thalach,'slope':slope,'restecg':restecg,'chol':chol,'trestbps':trestbps,'fbs':fbs,'oldpeak':oldpeak}

    input_query = np.array([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exe, oldpeak, slope]])

    result = model.predict(input_query)[0]

    return jsonify({'hearth_disease': str(result)})



if __name__ == '__main__':
    app.run(debug=True)