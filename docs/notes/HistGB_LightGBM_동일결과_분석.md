# HistGradientBoosting과 LightGBM이 완전히 동일한 결과를 낸 이유

## 관찰

물성 회귀 모델 비교(녹는점, 끓는점)에서 `HistGradientBoosting`과 `LightGBM`의 R², RMSE, MAE가 소수점 6자리까지 같았다.

| 모델 | R²(mp) | RMSE(mp) | MAE(mp) |
|---|---|---|---|
| HistGradientBoosting | 0.631928 | 59.628855 | 43.970805 |
| LightGBM | 0.631928 | 59.628855 | 43.970805 |

두 데이터셋 모두에서 같은 현상이 나타났다. 먼저 코드 실수(딕셔너리 참조 오류, import 실패 등)를 의심했고, 다음을 확인해 배제했다.

- `models['LightGBM'] is models['HistGradientBoosting']` → `False` (서로 다른 객체)
- 내부 모델 클래스 확인 → 각각 `lightgbm.sklearn.LGBMRegressor`, `sklearn...HistGradientBoostingRegressor`로 정상
- `lightgbm` 라이브러리 정상 import 확인 (버전 4.7.0)

## 원인 1. 기본 하이퍼파라미터가 거의 같다

scikit-learn의 `HistGradientBoostingRegressor`는 LightGBM의 히스토그램 기반 그래디언트 부스팅을 참고해 만든 구현이라, 기본값이 다음과 같이 대응된다.

| 파라미터 | HistGradientBoostingRegressor | LGBMRegressor |
|---|---|---|
| 리프 개수 | `max_leaf_nodes=31` | `num_leaves=31` |
| 학습률 | `learning_rate=0.1` | `learning_rate=0.1` |
| 트리(부스팅 라운드) 개수 | `max_iter=100` | `n_estimators=100` |
| 리프당 최소 샘플 수 | `min_samples_leaf=20` | `min_child_samples=20` |
| L2 규제 | `l2_regularization=0.0` | `reg_lambda=0.0` |
| 히스토그램 bin 수 | `max_bins=255` | `max_bin=255` |

## 원인 2. 이진 피처라 분할 지점이 사실상 하나뿐이다

히스토그램 기반 부스팅은 피처마다 어느 값을 기준으로 나눌지 탐색한다. 연속값 피처는 분할 후보가 많지만, Morgan fingerprint는 0/1 이진값이라 분할 지점이 0과 1 사이 하나뿐이다. 그래서 구현이 달라도 매 단계에서 같은 분할을 고르게 된다.

## 검증: 합성 데이터로 재현

같은 조건(이진 피처, 기본 하이퍼파라미터, 같은 `random_state`)을 합성 데이터로 만들어 재현했다.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(0)
n_samples, n_features = 2000, 500
X = np.random.randint(0, 2, size=(n_samples, n_features)).astype(float)  # 이진 피처
true_weights = np.random.randn(n_features)
y = X @ true_weights + np.random.randn(n_samples) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_train, y_train)
pred_hgb = hgb.predict(X_test)

lgb = LGBMRegressor(random_state=42, verbose=-1)
lgb.fit(X_train, y_train)
pred_lgb = lgb.predict(X_test)

print(np.allclose(pred_hgb, pred_lgb))          # True
print(np.max(np.abs(pred_hgb - pred_lgb)))       # 1.86e-07 (부동소수점 오차 수준)
```

두 모델의 예측값 차이는 최대 `1.86e-07`로 부동소수점 오차 수준이었다. 버그가 아니라 이진 피처와 같은 기본 하이퍼파라미터 조합에서 재현되는 현상이다.

## 결론

- 결과 자체는 신뢰할 수 있고, 이 조건에서는 두 모델을 사실상 같은 모델로 봐도 된다.
- 두 모델을 다르게 동작시키려면 `num_leaves`/`max_leaf_nodes`, `learning_rate`, `n_estimators`/`max_iter` 등을 다르게 설정해야 한다.
- 이후 연속값인 서술자(LogP, TPSA 등)를 fingerprint에 추가하자 두 모델의 결과가 조금씩 달라졌다. 원인 2가 맞았다는 근거다.
