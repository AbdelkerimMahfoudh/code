from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

model = joblib.load('prediction.model')

categorical_features = ["Project_name", "country"]
numerical_features = ["backers_count", "goal", "pledged", "Average Contribution"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])

    le = LabelEncoder()
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    prediction = model.predict(df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)