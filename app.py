from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        data = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(data)
        return render_template("index.html", prediction=prediction[0], user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
