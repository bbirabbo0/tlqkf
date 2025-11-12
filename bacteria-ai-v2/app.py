import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 모델 불러오기
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # 1) 들어온 값으로 DataFrame 만들기 (학습할 때 썼던 컬럼 이름 그대로!)
    X_input = pd.DataFrame([{
        "place_type": data["place_type"],
        "people_level": data["people_level"],
        "touch_level": data["touch_level"],
        "clean_level": data["clean_level"],
        "humidity_level": data["humidity_level"],
        "colonies": float(data["colonies"])
    }])

    # 2) 모델에 넣어서 예측
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0].max()

    return jsonify({
        "risk_label": str(pred),
        "confidence": round(float(prob), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
