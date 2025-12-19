import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)

rf = pickle.load(open("Models/random_forest_model.pkl", "rb"))
gb = pickle.load(open("Models/gradient_boosting_model.pkl", "rb"))
lr = pickle.load(open("Models/logistic_regression_model.pkl", "rb"))

def ensemble_predict(freq, signal):
    X = [[freq, signal]]
    preds = [
        rf.predict(X)[0],
        gb.predict(X)[0],
        lr.predict(X)[0],
    ]
    return Counter(preds).most_common(1)[0][0]

@app.route("/")
def health():
    return "RF ML Service is running"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        freq = float(data.get("frequency"))
        signal = float(data.get("signal"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid frequency or signal"}), 400

    result = ensemble_predict(freq, signal)
    return jsonify({"classification": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
