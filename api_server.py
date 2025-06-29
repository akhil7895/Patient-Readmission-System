# api_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load the trained ML model
model = joblib.load('readmission_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate that the required fields are present
        required_fields = ['age', 'num_procedures', 'num_medications', 'time_in_hospital']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field: {field}"}), 400

        # Extract features
        age = data['age']
        num_procedures = data['num_procedures']
        num_medications = data['num_medications']
        time_in_hospital = data['time_in_hospital']

        # Prepare input for model
        input_features = np.array([[age, num_procedures, num_medications, time_in_hospital]])
        
        # Predict using model
        prediction = model.predict(input_features)

        # Prepare result
        result = 'Readmitted' if prediction[0] == 1 else 'Not Readmitted'

        # Send back the result
        return jsonify({'prediction': result})

    except Exception as e:
        # Return any error that occurs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
