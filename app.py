from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
# Ensure the model file exists
model_path = 'Froud_detection.joblib'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please ensure the model is trained and saved.")
    model = None
else:
    model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # The model expects 30 features: Time, V1-V28, Amount
        features = []
        
        # Time
        features.append(float(data.get('Time', 0)))
        
        # V1 to V28
        for i in range(1, 29):
            key = f'V{i}'
            features.append(float(data.get(key, 0)))
            
        # Amount
        features.append(float(data.get('Amount', 0)))
        
        # Convert to numpy array and reshape for prediction
        final_features = np.array([features])
        
        # Predict
        prediction = model.predict(final_features)
        output = int(prediction[0])
        
        result_text = "Fraudulent Transaction" if output == 1 else "Normal Transaction"
        
        return jsonify({
            'prediction_text': result_text,
            'is_fraud': output == 1
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
