from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("liver_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['age']),
            int(request.form['gender']),
            float(request.form['bmi']),
            float(request.form['alcohol']),
            int(request.form['smoking']),
            int(request.form['genetic']),
            float(request.form['activity']),
            int(request.form['diabetes']),
            int(request.form['hypertension']),
            float(request.form['liver_function']),
        ]

        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]

        result_text = "🛑 Positive: Liver Cirrhosis Detected" if prediction == 1 else "✅ Negative: No Liver Cirrhosis"

        # Additional message/advice
        advice = "Please consult a liver specialist immediately for further diagnosis and management." if prediction == 1 else "Keep maintaining a healthy lifestyle and regular checkups."

        return render_template('result.html', prediction=result_text, advice=advice)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}", advice="Try again with valid inputs.")

if __name__ == '__main__':
    app.run(debug=True)
