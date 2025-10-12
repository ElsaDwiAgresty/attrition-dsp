from flask import Flask, render_template, request
import pandas as pd
import os

# Asumsikan fungsi-fungsi ini sudah didefinisikan di file lain atau di sini
from model_util import set_features, load_model, generate_random_features

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard_view.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    features_names = set_features() 
    model = load_model()  
    features = generate_random_features() if request.method == "GET" else None  
    prediction = None

    if request.method == "POST":
        # Mengambil data dari form dan mengonversi ke integer
        features = {name: int(request.form.get(name, 0)) for name in features_names}
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]
        prediction = "Yes" if pred == 1 else "No" if pred == 0 else str(pred)

    return render_template('predict_view.html', features=features, prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)