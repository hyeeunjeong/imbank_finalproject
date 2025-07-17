from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, static_folder='.')
CORS(app)

# ✅ 모델 클래스 정의
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

# ✅ 모델 로드
with open('credit_score_predictor/credit_score_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# ✅ HTML 시작 페이지
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# ✅ 정적 파일 서빙
@app.route('/<path:path>')
def serve_static_file(path):
    return send_from_directory('.', path)

# ✅ API 라우팅
@app.route('/predict_score')
def predict_score():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId가 필요합니다.'}), 400

    try:
        score = model.predict_by_id(user_id)
    except Exception as e:
        print(f"[ERROR] Predict failed for userId={user_id}: {e}")  # 터미널에 에러 출력
        return jsonify({'error': str(e)}), 500

    return jsonify({'score': score})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
