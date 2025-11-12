import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 데이터 불러오기
data = pd.read_csv("school_bacteria_data.csv", encoding="cp949")


X = data[["place_type", "people_level", "touch_level",
          "clean_level", "humidity_level", "colonies"]]
y = data["risk_label"]

# 전처리
cat_cols = ["place_type", "people_level", "touch_level",
            "clean_level", "humidity_level"]
num_cols = ["colonies"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

# 학습
model.fit(X, y)

# 모델 저장
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 모델 저장 완료: risk_model.pkl")
