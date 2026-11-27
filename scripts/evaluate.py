"""예측 파일(pred_*.csv / pred_*.csv.gz)을 모아 지표 CSV 저장.
출력 (results/):
  metrics_overall.csv            모델별 24개월 전체 MAE·MASE·WQL (+ 추론 시간)
  metrics_by_category.csv        카테고리 × 모델
  metrics_monthly.csv            달 × 모델 (예측 24개월 각각)
  metrics_monthly_by_category.csv  카테고리 × 달 × 모델
사용법: python scripts/evaluate.py [--data ...] [--pred results/predictions] [--out results]
"""
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import spendradar_eval as se

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(ROOT / 'data' / ('tabformer_monthly_long.csv' if (ROOT / 'data' / 'tabformer_monthly_long.csv').exists() else 'tabformer_monthly_long.csv.gz')))
ap.add_argument('--pred', default=str(ROOT / 'results' / 'predictions'))
ap.add_argument('--out', default=str(ROOT / 'results'))
args = ap.parse_args()

d = se.load_data(args.data)
print(d.describe())
per, overall, by_cat = se.evaluate(d, args.pred)
monthly, monthly_cat = se.evaluate_by_month(d, args.pred)

o = overall[['MAE', 'MASE', 'WQL']].sort_values('MASE')
rp = os.path.join(args.pred, 'runtime.csv')
if os.path.exists(rp):
    o = o.join(pd.read_csv(rp, encoding='utf-8-sig').set_index('model')[['추론 시간(초)', '파라미터']])
o = o.round({'MAE': 2, 'MASE': 6, 'WQL': 6})
o.to_csv(os.path.join(args.out, 'metrics_overall.csv'), encoding='utf-8-sig')
by_cat[['MAE', 'MASE', 'WQL']].round(6).to_csv(os.path.join(args.out, 'metrics_by_category.csv'), encoding='utf-8-sig')
monthly[['model', 'month', 'MAE', 'MASE', 'WQL']].round(6).to_csv(os.path.join(args.out, 'metrics_monthly.csv'), index=False, encoding='utf-8-sig')
monthly_cat[['category', 'model', 'month', 'MAE', 'MASE', 'WQL']].round(6).to_csv(
    os.path.join(args.out, 'metrics_monthly_by_category.csv'), index=False, encoding='utf-8-sig')
print(o.to_string())
print(se.month_tables(monthly)['MASE'].round(3).to_string())
