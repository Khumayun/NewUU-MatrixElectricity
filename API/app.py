from flask import Flask, request, render_template
import pickle






model = pickle.load(open('Matrix G.ipynb','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')




app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    data = pd.read_csv(uploaded_file)


    return "File uploaded and processed successfully!"



@app.route('/predict', methods=['POST'])
def predict():
    feature0 = float(request.form['feature0'])
    prediction = model.predict_next_year([[feature0]])
    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
