# Turbidity Forecast GUI

탁도 예측/경보를 위한 데스크톱 GUI 프로그램입니다.

- 예측 수평선(horizon): **1일, 2일, 3일**
- 경보 기준:
  - `WATCH`: \(P(\text{Turbidity} > 10)\) >= WATCH cut
  - `WARNING`: \(P(\text{Turbidity} > 20)\) >= WARNING cut
- 지원 모델: **LightGBM, CatBoost, XGBoost**
- 앙상블: **mean(평균)** / **weighted(가중평균)**

---

## 1. 실행 방법

## 권장 환경
- Windows
- Python 3.12 가상환경 (`.venv-312`)

## 실행 명령
```powershell
& "d:\dev\GPT_Turbidity\.venv-312\Scripts\python.exe" "d:\dev\GPT_Turbidity\turbidity_gui_app.py"
```

또는 가상환경 활성화 후:
```powershell
python .\turbidity_gui_app.py
```

---

## 2. GUI 사용 순서

1. **입력 파일**(`flow.csv`, `met_data.csv`, `water_quality.csv`) 선택  
2. **모델 선택(학습)** 체크  
3. 필요 시 **앙상블 설정**(사용 여부, 방식, 가중치) 설정  
4. **모델 학습/평가 실행** 클릭  
5. **표시 옵션**에서 비교 모델/Horizon/Split/Threshold 선택 후 결과 확인  
6. 필요 시 **신규 입력 + 3일 예측**에서 최신값 직접 입력 후  
   **모델 실행 (미래 3일 예측)** 클릭  
7. **결과 CSV 저장**으로 결과 파일 저장

---

## 3. 입력 CSV 형식

## 3.1 flow.csv (일 단위 유량)
- 필수 컬럼:
  - `date`: 날짜
  - `flow`: 유량(숫자)

예시:
```csv
date,flow
2014-01-01,12.3
2014-01-02,11.8
```

## 3.2 met_data.csv (일 단위 기상)
- 필수 컬럼:
  - `Date`: 날짜
  - `Temp`: 기온
  - `Precipitation`: 강수량

예시:
```csv
Date,Temp,Precipitation
2014-01-01,1.2,0.0
2014-01-02,0.8,5.1
```

## 3.3 water_quality.csv (수질 관측)
- 필수 컬럼:
  - `date`: 관측 시각(시간 단위 허용)
  - `turb`: 탁도
  - `pH`
  - `alk`
  - `ec`
  - `wt`

예시:
```csv
date,turb,pH,alk,ec,wt
2014-01-01 00:00:00,3.2,7.3,48,210,5.0
2014-01-01 01:00:00,3.1,7.3,48,209,4.9
```

> `water_quality.csv`는 내부에서 **일평균**으로 집계되어 사용됩니다.

---

## 4. 예측 모델 간단 설명

- **LightGBM**  
  Gradient Boosting Decision Tree 기반. 속도/성능 균형이 좋고 시계열 lag 피처와 결합이 좋음.

- **CatBoost**  
  범주형 처리에 강점이 있지만, 수치형 중심 문제에서도 일반화 성능이 안정적인 경우가 많음.

- **XGBoost**  
  규제 옵션이 풍부하고 보수적인 학습으로 과적합 제어에 유리.

---

## 5. 앙상블 예측 방법

## 5.1 mean
모델 확률/예측값을 단순 평균:

- `turb_pred_ens = (turb_pred_lgbm + turb_pred_cat + turb_pred_xgb) / N`
- `p_gt_10_ens`, `p_gt_20_ens`도 동일하게 평균

초기 학습/평가 실행 시 앙상블은 기본적으로 이 방식으로 계산됩니다.

## 5.2 weighted
사용자 가중치(또는 추천 가중치)를 사용한 가중 평균:

- `ens = w_lgbm * lgbm + w_cat * cat + w_xgb * xgb`
- 내부적으로 가중치는 합이 1이 되도록 정규화되어 사용됩니다.

## 5.3 성능기반 가중치 추천
`성능기반 가중치 추천` 버튼은 **test 구간 평균 F1 최대화**를 목표로 가중치를 탐색해 적용합니다.

---

## 6. 화면에서 확인 가능한 결과

- **시계열 그래프**  
  실측 vs 예측, train/test 분리 표시, 모델별 비교, 미래 3일 예측선 표시

- **Confusion Matrix 테이블**  
  모델별/수평선별/threshold별 TP/FP/TN/FN, Accuracy/Precision/Recall/F1

- **3일 예측 결과 테이블**  
  `input_date`, `target_date`, `horizon_days`, `turb_pred`, `p_gt_10`, `p_gt_20`, `alarm`

---

## 7. 저장 파일

`output_alarm` 폴더에 저장:

- `gui_timeseries_predictions.csv`
- `gui_confusion_matrix.csv`
- `gui_3day_forecast.csv` (3일 예측 실행 시)

---

## 8. 참고 (CLI 스크립트)

GUI 외에 CLI도 사용 가능:

```powershell
python .\daily_turbidity_alarm.py --flow .\flow.csv --met .\met_data.csv --wq .\water_quality.csv
```

