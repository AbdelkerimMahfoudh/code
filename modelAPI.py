from flask import Flask, request, jsonify
import joblib

model = joblib.load("prediction.model")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
  # Get user input data from request
  data = request.get_json()
  
  # Make prediction
  prediction = model.predict_proba(pd.DataFrame([preprocessed_data]))[:, 1][0]
  success_proba = round(prediction * 100, 2)
  
  # Return prediction as JSON
  return jsonify({"success_probability": success_proba})