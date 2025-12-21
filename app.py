import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)

rf = joblib.load("Models/random_forest_model.pkl")
gb = joblib.load("Models/gradient_boosting_model.pkl")
lr = joblib.load("Models/logistic_regression_model.pkl")

LABEL_MAP = {
    0: "LONG TERM DAMAGES",
    1: "SHORT TERM DAMAGES",
    2: "HARMFUL",
    3: "SAFE"
}

def rule_based_classify(freq, signal):
    if 70e6 <= freq <= 113466496:
        if signal >= -50:
            return "SAFE"
        if signal >= -85:
            return "SHORT TERM DAMAGES"
        return "LONG TERM DAMAGES"

    if 113466496 < freq <= 160000000:
        if signal >= -55:
            return "SAFE"
        if signal >= -90:
            return "SHORT TERM DAMAGES"
        return "LONG TERM DAMAGES"

    if freq > 160000000:
        if signal >= -60:
            return "SAFE"
        if signal >= -95:
            return "SHORT TERM DAMAGES"
        return "LONG TERM DAMAGES"

    return None

def ensemble_predict(freq, signal):
    X = [[freq, signal]]
    preds = [
        rf.predict(X)[0],
        gb.predict(X)[0],
        lr.predict(X)[0],
    ]
    final_pred = Counter(preds).most_common(1)[0][0]
    if hasattr(final_pred, "item"):
        final_pred = final_pred.item()
    return LABEL_MAP.get(final_pred, "UNKNOWN")

@app.route("/")
def health():
    return "RF ML Service is running"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"classification": "UNKNOWN"})

    try:
        freq = float(data.get("frequency"))
        signal = float(data.get("signal"))
    except:
        return jsonify({"classification": "UNKNOWN"})

    rule = rule_based_classify(freq, signal)
    if rule:
        return jsonify({"classification": rule})

    return jsonify({"classification": ensemble_predict(freq, signal)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
