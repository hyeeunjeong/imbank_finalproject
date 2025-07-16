# (1) 클래스 정의 먼저 포함
import pandas as pd

class CreditScorePredictor:
    def __init__(self, clf_model, reg_model, feature_names, data: pd.DataFrame, id_col: str):
        self.clf = clf_model
        self.reg = reg_model
        self.features = feature_names
        self.data = data
        self.id_col = id_col

    def predict(self, row: pd.Series) -> float:
        X_input = row[self.features].values.reshape(1, -1)
        is_low_credit = self.clf.predict(X_input)[0]
        return 250.0 if is_low_credit == 1 else round(self.reg.predict(X_input)[0], 1)

    def predict_by_id(self, cust_id) -> float:
        customer_row = self.data[self.data[self.id_col] == cust_id]
        if customer_row.empty:
            raise ValueError(f"ID {cust_id}에 해당하는 고객 데이터를 찾을 수 없습니다.")
        row = customer_row.iloc[0]
        return self.predict(row)

# (2) 모델 불러오기
import pickle
with open('credit_score_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# (3) 예측
cust_id = model.data[model.id_col].iloc[0]
score = model.predict_by_id(cust_id)
print(f"예측 점수: {score}")