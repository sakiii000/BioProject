from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允許前端跨域存取

@app.route("/")
def home():
    return "Flask server is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sequence = data.get("sequence", "")
    mail = data.get("mail", "")
    
    # 模擬預測邏輯（實際請替換）
    prediction = "SNARE" if "Q" in sequence else "non-SNARE"
    
    return jsonify({"prediction": prediction})
