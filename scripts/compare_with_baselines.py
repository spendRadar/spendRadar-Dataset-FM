"""파운데이션 모델 예측 + Baseline 예측을 한 번에 평가해 비교표 저장.
Baseline 예측은 spendRadar-Dataset-Baseline 레포의 results/predictions 를 사용.
사용법:
  git clone https://github.com/spendRadar/spendRadar-Dataset-Baseline.git ../spendRadar-Dataset-Baseline
  python scripts/compare_with_baselines.py --baseline-pred ../spendRadar-Dataset-Baseline/results/predictions
출력: results/comparison_overall.csv, comparison_by_category.csv, comparison_monthly.csv
"""
import argparse, os, shutil, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import spendradar_eval as se

FM = ['TimesFM2.5', 'Moirai2.0', 'Chronos2', 'FinCast']
ap = argparse.ArgumentParser()
ap.add_argument('--data', default=str(ROOT / 'data' / 'tabformer_monthly_long.csv.gz'))
ap.add_argument('--pred', default=str(ROOT / 'results' / 'predictions'))
ap.add_argument('--baseline-pred', required=True)
ap.add_argument('--out', default=str(ROOT / 'results'))
args = ap.parse_args()

tmp = tempfile.mkdtemp()
rts = []
for d in (args.pred, args.baseline_pred):
    for f in se._pred_files(d):
        shutil.copy(f, tmp)
    rp = os.path.join(d, 'runtime.csv')
    if os.path.exists(rp):
        rts.append(pd.read_csv(rp, encoding='utf-8-sig'))
data = se.load_data(args.data)
per, overall, by_cat = se.evaluate(data, tmp)
monthly, _ = se.evaluate_by_month(data, tmp)
rt = pd.concat(rts).drop_duplicates('model', keep='last').set_index('model')

o = overall[['MAE', 'MASE', 'WQL']].sort_values('MASE')
o.insert(0, '구분', ['파운데이션 모델' if m in FM else 'Baseline' for m in o.index])
o['MASE 순위'] = o.MASE.rank().astype(int)
o['WQL 순위'] = o.WQL.rank().astype(int)
o = o.join(rt[['추론 시간(초)', '파라미터']])
o.round({'MAE': 2, 'MASE': 6, 'WQL': 6}).to_csv(os.path.join(args.out, 'comparison_overall.csv'), encoding='utf-8-sig')
by_cat[['MAE', 'MASE', 'WQL']].round(6).to_csv(os.path.join(args.out, 'comparison_by_category.csv'), encoding='utf-8-sig')
monthly[['model', 'month', 'MAE', 'MASE', 'WQL']].round(6).to_csv(os.path.join(args.out, 'comparison_monthly.csv'), index=False, encoding='utf-8-sig')
print(o.round(4).to_string())
mt = se.month_tables(monthly)['MASE'][o.index]
print('\n달별 MASE 1위 횟수:', mt.idxmin(axis=1).value_counts().to_dict())
print('\n카테고리별 MASE 1위:', by_cat.MASE.unstack('model').idxmin(axis=1).to_dict())
shutil.rmtree(tmp)
