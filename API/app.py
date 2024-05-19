from flask import Flask, request, render_template
import joblib

import tensorflow

file_name = joblib.dump(tensorflow, "Sequencial.joblib")

model = joblib.load(file_name)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    prediction = model.predict([[feature1, feature2]])
    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
