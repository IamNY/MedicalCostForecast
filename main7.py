import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 데이터 로드
data = pd.read_csv('data/insurance.csv')

# 데이터 크기를 줄이기 위해 샘플링
data_sample = data.sample(frac=1, random_state=60)

# 일부 데이터의 의료비용을 결측값으로 설정하기 전에 저장
np.random.seed(60)
missing_rate = 0.2
n_missing = int(len(data_sample) * missing_rate)
missing_indices = np.random.choice(data_sample.index, n_missing, replace=False)
original_values_sample = data_sample.loc[missing_indices, 'MedicalCost']

# 의료비용을 결측값으로 설정
data_sample.loc[missing_indices, 'MedicalCost'] = np.nan

# 결측값이 있는 데이터와 없는 데이터 분리
data_train_sample = data_sample.dropna(subset=['MedicalCost'])
data_test_sample = data_sample[data_sample['MedicalCost'].isna()]

# 특성과 타겟 분리
X_train_sample = data_train_sample.drop('MedicalCost', axis=1)
y_train_sample = data_train_sample['MedicalCost']
X_test_sample = data_test_sample.drop('MedicalCost', axis=1)

# 범주형 변수 인코딩
X_train_sample = pd.get_dummies(X_train_sample, drop_first=True)
X_test_sample = pd.get_dummies(X_test_sample, drop_first=True)

# 스케일링
scaler = StandardScaler()
X_train_sample[['Age', 'BMI', 'Children']] = scaler.fit_transform(X_train_sample[['Age', 'BMI', 'Children']])
X_test_sample[['Age', 'BMI', 'Children']] = scaler.transform(X_test_sample[['Age', 'BMI', 'Children']])

# 모델 리스트
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=60),
    'XGBoost': xgb.XGBRegressor(random_state=60),
    'Gradient Boosting': GradientBoostingRegressor(random_state=60),
    'Ridge': Ridge()
}

# 결과 저장용 데이터프레임
results = pd.DataFrame(columns=['Model', 'MSE', 'R2'])

# 모델 학습 및 예측
actual_vs_predicted_list = []

for model_name, model in models.items():
    model.fit(X_train_sample, y_train_sample)
    y_pred_sample = model.predict(X_test_sample)

    # 평가 지표 계산
    mse_sample = mean_squared_error(original_values_sample, y_pred_sample)
    r2_sample = r2_score(original_values_sample, y_pred_sample)

    # 결과 저장
    results = pd.concat([results, pd.DataFrame({
        'Model': [model_name],
        'MSE': [f"{mse_sample:.6f}"],  # 소수점 이하 6자리까지 숫자로 표기
        'R2': [r2_sample]
    })], ignore_index=True)

    # 실제 값과 예측 값 저장
    actual_vs_predicted = pd.DataFrame({
        'Model': model_name,
        'Actual': original_values_sample.values,
        'Predicted': y_pred_sample
    })
    actual_vs_predicted_list.append(actual_vs_predicted)

# 결과 출력
print(results)

# 각 모델별 실제 값과 예측 값 출력
for df in actual_vs_predicted_list:
    print(f"\nModel: {df['Model'].iloc[0]}")
    print(df[['Actual', 'Predicted']].head())
