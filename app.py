from flask import Flask, request, jsonify, render_template
import logging
import pickle

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Lazy load model and scaler
model = None
scaler = None

@app.route('/')
def home():
    logging.info("Root endpoint accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    try:
        # Load model and scaler if not already loaded
        if model is None or scaler is None:
            logging.info("Loading model and scaler")
            model = pickle.load(open('model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))

        # Get JSON data from the request
        data = request.get_json()
        year = int(data.get("year"))
        month = int(data.get("month"))

        # Validate year and month inputs
        if not (1900 <= year <= 2100):
            return jsonify({"error": "Year must be between 1900 and 2100"}), 400
        if not (1 <= month <= 12):
            return jsonify({"error": "Month must be between 1 and 12"}), 400

        # Prepare data for prediction
        month_index = year * 12 + month
        month_index_scaled = scaler.transform([[month_index]])

        # Make prediction
        prediction = model.predict(month_index_scaled)
        logging.info(f"Prediction for {year}-{month}: {prediction[0]}")
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logging.info("Health endpoint accessed")
    return 'OK', 200

if __name__ == '__main__':
    app.run(debug=True)