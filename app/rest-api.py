from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import joblib
import traceback
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/prediction', methods=['POST'])
def predict():
    try:
        json_ = request.json
        print(json_)
        query = pd.DataFrame([json_])
        predict_array = model.predict_proba(query)
        predict = round(predict_array[0][1], 2)
        print('Predicted survival chance is ' + str(predict * 100) + '%') 
        return jsonify({'prediction': predict})
    except:
        return jsonify({'trace': traceback.format_exc()})

@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')

if __name__ == '__main__':
    port = 80 
    model = joblib.load('titanic-predict.pkl') 
    print ('Model loaded')
    app.run(host='0.0.0.0', port=port, debug=True)
