from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the front-end

# Load the model (ensure you have a pre-trained model saved as 'model.pkl')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for home
@app.route('/')
def home():
    return "Fraud Detection API is running!"

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the input data is valid
        if not request.json or not isinstance(request.json, list):
            return jsonify({'error': 'Invalid input format. Expecting a JSON array of objects.'}), 400

        # Convert input JSON to DataFrame
        input_data = pd.DataFrame(request.json)
        print("Received Input:", input_data)  # Debugging

        # Ensure correct column name (no typo)
        input_data.rename(columns={'bal_of_recepient_before_transaction': 'bal_of_recepient_before_transaction'}, inplace=True)

        # Ensure input columns match the model's expected features
        required_columns = [
            'step', 'amount', 'bal_before_transaction', 'bal_after_transaction',
            'bal_of_recepient_before_transaction', 'bal_of_receipient_after_transaction',
            'is_flagged_fraud', 'type_CASH_IN', 'type_CASH_OUT',
            'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]

        # Check and add any missing columns
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Fill missing columns with 0

        # Predict using the loaded model
        predictions = model.predict(input_data)
        response = {'predictions': predictions.tolist()}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
