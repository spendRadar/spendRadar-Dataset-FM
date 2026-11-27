"""IBM TabFormer 원본(card_transaction.v1.csv) → 월간 카테고리별 지출 시계열.
사용법: python src/prep_tabformer_monthly.py <card_transaction.v1.csv> <출력 폴더>
"""
import sys, pandas as pd, numpy as np
src, out = sys.argv[1:3]
cmap = {1711: '주거/통신', 3000: '문화/여가', 3001: '문화/여가', 3005: '문화/여가', 3006: '문화/여가', 3007: '문화/여가', 3008: '문화/여가', 3009: '문화/여가', 3058: '문화/여가', 3066: '문화/여가', 3075: '문화/여가', 3132: '문화/여가', 3144: '문화/여가', 3174: '문화/여가', 3256: '문화/여가', 3260: '문화/여가', 3359: '교통/차량', 3387: '교통/차량', 3389: '교통/차량', 3390: '교통/차량', 3393: '교통/차량', 3395: '교통/차량', 3405: '교통/차량', 3504: '문화/여가', 3509: '문화/여가', 3596: '문화/여가', 3640: '문화/여가', 3684: '문화/여가', 3722: '문화/여가', 3730: '문화/여가', 3771: '문화/여가', 3775: '문화/여가', 3780: '문화/여가', 4111: '교통/차량', 4112: '교통/차량', 4121: '교통/차량', 4131: '교통/차량', 4214: '기타', 4411: '문화/여가', 4511: '문화/여가', 4722: '문화/여가', 4784: '교통/차량', 4814: '주거/통신', 4899: '주거/통신', 4900: '주거/통신', 5045: '쇼핑/패션', 5094: '쇼핑/패션', 5192: '문화/여가', 5193: '쇼핑/패션', 5211: '주거/통신', 5251: '주거/통신', 5261: '주거/통신', 5300: '쇼핑/패션', 5310: '쇼핑/패션', 5311: '쇼핑/패션', 5411: '식비/카페', 5499: '식비/카페', 5533: '교통/차량', 5541: '교통/차량', 5621: '쇼핑/패션', 5651: '쇼핑/패션', 5655: '쇼핑/패션', 5661: '쇼핑/패션', 5712: '주거/통신', 5719: '주거/통신', 5722: '주거/통신', 5732: '쇼핑/패션', 5733: '쇼핑/패션', 5812: '식비/카페', 5813: '식비/카페', 5814: '식비/카페', 5815: '문화/여가', 5816: '문화/여가', 5912: '의료', 5921: '식비/카페', 5932: '쇼핑/패션', 5941: '쇼핑/패션', 5942: '문화/여가', 5947: '쇼핑/패션', 5970: '쇼핑/패션', 5977: '쇼핑/패션', 6300: '기타', 7011: '문화/여가', 7210: '주거/통신', 7230: '쇼핑/패션', 7276: '기타', 7349: '주거/통신', 7393: '기타', 7531: '교통/차량', 7538: '교통/차량', 7542: '교통/차량', 7549: '교통/차량', 7801: '문화/여가', 7802: '문화/여가', 7832: '문화/여가', 7922: '문화/여가', 7995: '문화/여가', 7996: '문화/여가', 8011: '의료', 8021: '의료', 8041: '의료', 8043: '의료', 8049: '의료', 8062: '의료', 8099: '의료', 8111: '기타', 8931: '기타', 9402: '기타'}
def cat(m):
    if m in cmap: return cmap[m]
    if 3000 <= m <= 3299: return '문화/여가'
    if 3351 <= m <= 3500: return '교통/차량'
    if 3501 <= m <= 3999: return '문화/여가'
    return None
cols = ['User','Year','Month','Amount','MCC','Errors?','Is Fraud?']
df = pd.read_csv(src, usecols=cols, dtype={'User':'int16','Year':'int16','Month':'int8','MCC':'int16'})
n0 = len(df)
START, END = pd.Period('2010-03','M'), pd.Period('2020-02','M')   # 최근 10년 (원본 마지막 달 = 2020-02)
TEST = START + int(120 * 0.8)                                      # 8:2 분할 → 입력 96개월, 예측 24개월
df['ym'] = pd.PeriodIndex.from_fields(year=df.Year, month=df.Month, freq='M')
df = df[(df.ym >= START) & (df.ym <= END)]
n_win = len(df)
df = df[(df['Is Fraud?'] == 'No') & df['Errors?'].isna() & (df.MCC != 4829)]
df['amount'] = df.Amount.str.replace('$', '', regex=False).astype(float)
df = df[df.amount > 0]
mm = {m: cat(m) for m in df.MCC.unique()}
miss = [m for m, c in mm.items() if c is None]; assert not miss, miss
df['category'] = df.MCC.map(mm)
months = pd.period_range(START, END, freq='M')
g = df.groupby('User')
first, last, cnt = g.ym.min(), g.ym.max(), g.size()
# 첫 3개월·마지막 3개월에 모두 거래가 있고, 10년간 거래 1,000건 이상인 사용자
users = first[(first <= START + 2) & (last >= END - 2) & (cnt >= 1000)].index
df = df[df.User.isin(users)]
cats = ['식비/카페','주거/통신','쇼핑/패션','교통/차량','문화/여가','의료','기타']
w = df.groupby(['User','category','ym']).amount.sum()
w = w.reindex(pd.MultiIndex.from_product([users, cats, months], names=['User','category','ym']), fill_value=0).reset_index()
tot = w.groupby(['User','ym']).amount.sum().reset_index(); tot['category'] = '전체'
w = pd.concat([w, tot], ignore_index=True)
w['dataset'] = 'tabformer'; w['entity_id'] = 'U' + w.User.astype(str).str.zfill(4)
w['unique_id'] = 'tabformer_' + w.entity_id + '_' + w.category
w['freq'] = 'M'; w['currency'] = 'USD'
w['ds'] = w.ym.dt.start_time.dt.strftime('%Y-%m-%d')
w['y'] = w.amount.round(2)
w['split'] = np.where(w.ym < TEST, 'train', 'test')
cols_out = ['dataset','unique_id','entity_id','category','freq','currency','ds','y','split']
long = w[cols_out].sort_values(['entity_id','category','ds'])
long.to_csv(f'{out}/tabformer_monthly_long.csv', index=False, encoding='utf-8-sig')
rng = np.random.default_rng(42); s = sorted(rng.choice(long.entity_id.unique(), 100, replace=False))
long[long.entity_id.isin(s)].to_csv(f'{out}/tabformer_monthly_long_sample100.csv', index=False, encoding='utf-8-sig')
c = long[long.category != '전체']
st = c.groupby('category').y.agg(mean='mean', zero=lambda v: (v == 0).mean() * 100).round(1)
print(f'원본 {n0:,} / 기간 내 {n_win:,} / 정제 {len(df):,}건, 사용자 {len(users)}명, 시계열 {long.unique_id.nunique():,}개, 월 {len(months)} (train {(months < TEST).sum()}, test {(months >= TEST).sum()})')
print(st.to_string())
