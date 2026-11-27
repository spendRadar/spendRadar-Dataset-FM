# SpendRadar 시계열 파운데이션 모델 평가

SpendRadar(가천대 AI P-실무프로젝트) 지출 예측에 쓸 모델을 고르기 위한 **시계열 파운데이션 모델 4종 zero-shot 평가** 코드·예측값·지표.
TabFormer 월간 카테고리 지출 데이터로 24개월을 예측하고 MAE·MASE·WQL을 달마다 비교함.

- 데이터셋·Baseline 4종: [spendRadar-Dataset-Baseline](https://github.com/spendRadar/spendRadar-Dataset-Baseline)
- 두 레포는 같은 데이터(`data/tabformer_monthly_long.csv.gz`)와 같은 지표 코드(`src/spendradar_eval.py`)를 사용함

**결론: 서비스 모델로 TimesFM 2.5 선정.** Moirai 2.0과 정확도 공동 1위이고, 식비/카페 등 핵심 카테고리에서 1위이며, 상업적으로 쓸 수 있는 라이선스(Apache-2.0)임.

## 1. 평가 모델
| 모델 | 체크포인트 | 크기 | 가중치 라이선스 | 논문 |
|---|---|---|---|---|
| TimesFM 2.5 (Google) | google/timesfm-2.5-200m-pytorch | 200M | Apache-2.0 | Das et al., ICML 2024 (TimesFM 원 논문, 2.5는 같은 구조의 개선판) |
| Moirai 2.0 (Salesforce) | Salesforce/moirai-2.0-R-small | 11M | CC BY-NC 4.0 (비상업) | Liu et al., arXiv:2511.11698, 2025 |
| Chronos-2 (Amazon) | amazon/chronos-2 | 120M | Apache-2.0 | Ansari et al., "Chronos-2: From Univariate to Universal Forecasting", arXiv:2510.15821, 2025 |
| FinCast | Vincent05R/FinCast (v1.pth) | 1B | 확인 필요 | Zhu et al., CIKM 2025 |

- 모두 추가 학습 없이(zero-shot) 입력 96개월 → 24개월 예측, 분위수 0.1 ~ 0.9 출력

## 2. 데이터
- IBM TabFormer 합성 카드 거래 → 사용자 1,311명 × 카테고리 7개(+ 전체) 월간 지출, 2010-03 ~ 2020-02
- 입력 96개월(2010-03 ~ 2018-02), 예측 24개월(2018-03 ~ 2020-02), 시계열 10,488개
- 전처리 방법·통계는 [Baseline 레포](https://github.com/spendRadar/spendRadar-Dataset-Baseline) README 참고

## 3. 결과 (Baseline 포함, 7개 카테고리 기준, 낮을수록 좋음)
| 순위 | 모델 | 구분 | MAE ($) | MASE | WQL | 추론 시간 (Colab T4) |
|---|---|---|---|---|---|---|
| 1 | Moirai 2.0 | 파운데이션 모델 | 209.59 | 0.7328 | 0.2834 | 1.7초 |
| 2 | TimesFM 2.5 | 파운데이션 모델 | 209.44 | 0.7331 | 0.2835 | 19.9초 |
| 3 | Chronos-2 | 파운데이션 모델 | 210.34 | 0.7362 | 0.2847 | 13.0초 |
| 4 | HistQuantile | Baseline | 213.14 | 0.7392 | 0.2879 | 0.5초 (CPU) |
| 5 | AutoARIMA | Baseline | 218.55 | 0.7731 | 0.2922 | 2,842초 (CPU) |
| 6 | FinCast | 파운데이션 모델 | 244.31 | 0.8640 | 0.3372 | 83.7초 |
| 7 | Naive | Baseline | 289.43 | 1.0251 | 0.5236 | 3.8초 (CPU) |
| 8 | SeasonalNaive | Baseline | 292.43 | 1.0280 | 0.3952 | 3.8초 (CPU) |

- 상위 3개(Moirai 2.0, TimesFM 2.5, Chronos-2)는 차이가 매우 작음 (1위와 3위의 MASE 차이 0.0034)
- 세 모델 모두 Baseline 중 가장 좋은 HistQuantile보다 정확함
- FinCast는 금융 시장 데이터용 모델이라 개인 지출에서는 HistQuantile보다 떨어짐
- Chronos-2는 상위 두 모델과 차이가 작지만 1위인 카테고리는 없음

### 달별 결과 (`results/comparison_monthly.csv`)
- 24개월 중 23개월은 TimesFM 2.5(13개월) 또는 Moirai 2.0(10개월)이 MASE 1위, 두 모델의 달별 MASE 차이는 평균 0.002 (최대 0.007)
- 2020-02는 28일뿐인 달이라 모든 모델의 오차가 커지고 HistQuantile이 근소하게 1위

### 카테고리별 MASE 1위 (`results/comparison_by_category.csv`)
| 카테고리 | 1위 |
|---|---|
| 식비/카페, 쇼핑/패션, 주거/통신, 의료 | TimesFM 2.5 |
| 교통/차량, 전체 합계 | Moirai 2.0 |
| 기타, 문화/여가 | HistQuantile |

### TimesFM 2.5를 고른 이유
- 정확도는 Moirai 2.0과 사실상 같음 (MAE는 TimesFM이 더 낮음)
- 지출 비중이 큰 식비/카페·쇼핑/패션·주거/통신에서 1위
- Moirai 2.0 가중치는 비상업 라이선스(CC BY-NC 4.0)라 서비스에 쓰기 어려움, TimesFM은 Apache-2.0
- 추론 20초(1만 개 시계열 기준)로 서비스에 충분히 빠름

### 지표
| 지표 | 정의 | 단위 |
|---|---|---|
| MAE | 실제 월 지출과 예측 중앙값(q0.5)의 평균 절대 차이 | $ |
| MASE | MAE ÷ 입력 구간에서 '작년 같은 달 값' 예측(m=12)의 평균 오차. 1 미만이면 그보다 정확 | 없음 |
| WQL | 분위수 0.1 ~ 0.9 예측의 가중 분위수 손실 (2 × Σpinball ÷ Σ\|실제\|, 분위수 평균) | 없음 |

## 4. 폴더 구조
```
data/
  tabformer_monthly_long.csv.gz          전체 데이터 (Baseline 레포와 동일)
  tabformer_monthly_long_sample100.csv   100명 샘플
  tabformer_mcc_category_map.csv         MCC → 카테고리 매핑표
src/
  run_model.py                파운데이션 모델 예측 (timesfm | moirai | chronos2 | fincast)
  spendradar_eval.py          데이터 로드, 예측 저장, 지표 계산 (Baseline 레포와 공통)
  prep_tabformer_monthly.py   원본 → 월간 데이터 (데이터를 새로 만들 때만)
scripts/
  colab_setup.sh              Colab 패키지 설치 (+ 데이터 없으면 원본 다운로드·전처리)
  evaluate.py                 파운데이션 모델 지표 → results/metrics_*.csv
  compare_with_baselines.py   Baseline 예측과 합쳐 비교표 → results/comparison_*.csv
notebooks/
  run_all_colab.ipynb         Colab에서 전체 실행 (약 10분)
results/
  predictions/pred_<모델>.csv.gz   시계열별 24개월 분위수 예측 (unique_id, ds, q0.1 ~ q0.9)
  predictions/runtime.csv          모델별 추론 시간
  metrics_*.csv                    파운데이션 모델 4종 지표 (전체 / 카테고리 / 달 / 카테고리×달)
  comparison_*.csv                 Baseline 포함 8개 모델 비교 (전체 / 카테고리 / 달)
  colab_logs/                      Colab 실행 로그
  SpendRadar_FM_results.xlsx       요약·월별·카테고리별 비교표 (엑셀)
```

## 5. 실행 방법 (Colab T4 GPU)
`notebooks/run_all_colab.ipynb`를 Colab에서 열고 위에서부터 실행. 직접 실행하려면:
```bash
git clone https://github.com/spendRadar/spendRadar-Dataset-FM.git && cd spendRadar-Dataset-FM
bash scripts/colab_setup.sh
python src/run_model.py timesfm
python src/run_model.py moirai
python src/run_model.py chronos2
python src/run_model.py fincast
python scripts/evaluate.py

# Baseline과 비교
git clone https://github.com/spendRadar/spendRadar-Dataset-Baseline.git ../spendRadar-Dataset-Baseline
python scripts/compare_with_baselines.py --baseline-pred ../spendRadar-Dataset-Baseline/results/predictions
```
- 예측 결과는 `results/predictions/pred_<모델>.csv`로 저장됨 (저장소에는 용량 때문에 소수 둘째 자리 반올림 후 .csv.gz로 올림, 지표는 동일)
- 검증 환경: Colab T4, Python 3.13, timesfm 3.0.2, chronos-forecasting 2.x, uni2ts 2.0.0

## 6. 한계
- TabFormer는 합성 데이터라 실제 가계부와 지출 패턴이 다를 수 있음 → 팀원 가계부로 추가 검증 예정
- 예측 구간을 한 번(2018-03 ~ 2020-02)만 평가함
- Moirai 2.0은 small 모델만 사용, 모든 모델 파인튜닝 없이 평가
