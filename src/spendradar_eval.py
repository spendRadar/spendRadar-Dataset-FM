"""SpendRadar 시계열 파운데이션 모델 평가 공통 함수.

모든 모델 노트북이 이 파일을 불러와서
  1) 데이터 로드 (load_data)
  2) 예측 결과 저장 (save_predictions)
를 같은 방식으로 하고, 마지막 평가 노트북에서
  3) 지표 계산 (evaluate)
을 한다.

예측 파일 형식 (모델 공통): unique_id, ds, q0.1, q0.2, ..., q0.9
  - 점 예측(point)은 중앙값 q0.5를 사용
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{q}" for q in QUANTILES]
SEASON_BY_FREQ = {"W": 1, "M": 12}  # MASE 분모의 시차 m (GIFT-Eval·GluonTS 기준): 주간 1, 월간 12


# ---------------------------------------------------------------- 데이터
class EvalData:
    """train 구간(입력)과 test 구간(정답)을 시계열별로 정리한 묶음."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        train = df[df.split == "train"]
        test = df[df.split == "test"]

        self.ids: list[str] = sorted(df.unique_id.unique())
        tr = {k: g.y.to_numpy(dtype=np.float32) for k, g in train.groupby("unique_id", sort=True)}
        te = {k: g for k, g in test.groupby("unique_id", sort=True)}

        self.contexts: list[np.ndarray] = [tr[k] for k in self.ids]   # 모델 입력
        self.actuals: list[np.ndarray] = [te[k].y.to_numpy(dtype=np.float64) for k in self.ids]
        self.future_ds: list[np.ndarray] = [te[k].ds.to_numpy() for k in self.ids]

        lens = {len(a) for a in self.actuals}
        if len(lens) != 1:
            raise ValueError(f"test 길이가 시계열마다 다름: {lens}")
        self.horizon: int = lens.pop()
        self.freq: str = str(df.freq.iloc[0]) if "freq" in df.columns else "W"
        self.season: int = SEASON_BY_FREQ.get(self.freq, 1)
        self.meta = (
            df.drop_duplicates("unique_id")
            .set_index("unique_id")
            .loc[self.ids, ["dataset", "entity_id", "category", "currency"]]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def describe(self) -> str:
        ctx = {len(c) for c in self.contexts}
        return (
            f"시계열 {len(self):,}개 | 사용자 {self.meta.entity_id.nunique():,}명 | "
            f"입력 길이 {sorted(ctx)} | 예측 길이 {self.horizon} | 단위 {self.freq} | MASE 시차 m={self.season}"
        )


def load_data(path: str) -> EvalData:
    df = pd.read_csv(path, parse_dates=["ds"])
    need = {"unique_id", "ds", "y", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"필수 컬럼 없음: {need - set(df.columns)}")
    return EvalData(df)


# ---------------------------------------------------------------- 예측 저장
def save_predictions(data: EvalData, quantile_preds, model_name: str, out_dir: str,
                     elapsed_sec: float | None = None, n_params: str | None = None) -> str:
    """quantile_preds: (시계열 수, 예측 길이, 9) 배열. 마지막 축 순서는 QUANTILES.
    elapsed_sec: 예측에 걸린 시간(초). runtime.csv에 기록되어 평가표에 함께 표시됨."""
    if elapsed_sec is not None:
        os.makedirs(out_dir, exist_ok=True)
        rp = os.path.join(out_dir, "runtime.csv")
        rt = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame(columns=["model", "추론 시간(초)", "파라미터", "시계열 수"])
        rows = rt[rt.model != model_name].to_dict("records")
        rows.append({"model": model_name, "추론 시간(초)": round(float(elapsed_sec), 1),
                     "파라미터": n_params or "", "시계열 수": len(data)})
        pd.DataFrame(rows).to_csv(rp, index=False, encoding="utf-8-sig")
    q = np.asarray(quantile_preds, dtype=np.float64)
    expect = (len(data), data.horizon, len(QUANTILES))
    if q.shape != expect:
        raise ValueError(f"예측 shape {q.shape} != 기대 {expect}")
    if not np.isfinite(q).all():
        raise ValueError("예측값에 NaN/inf가 있음")
    out = pd.DataFrame(q.reshape(-1, len(QUANTILES)), columns=QCOLS)
    out.insert(0, "ds", np.concatenate(data.future_ds))
    out.insert(0, "unique_id", np.repeat(data.ids, data.horizon))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pred_{model_name}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장] {path}  ({len(data):,}개 시계열 × {data.horizon}기간)")
    return path


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ---------------------------------------------------------------- Baseline (statsforecast)
def run_hist_quantile(data: EvalData, out_dir: str):
    """HistQuantile Baseline: 과거(train) 월 지출의 분위수를 미래 모든 달에 그대로 사용.
    "평소 이 카테고리에 한 달에 이 정도 쓴다"는 가장 단순한 가계부식 예측."""
    import time
    t0 = time.time()
    q = np.stack([np.tile(np.quantile(c, QUANTILES), (data.horizon, 1)) for c in data.contexts])
    return save_predictions(data, q, "HistQuantile", out_dir, elapsed_sec=time.time() - t0, n_params="통계 (CPU)")


def run_baselines(data: EvalData, out_dir: str, models=None, n_jobs: int = -1):
    """통계 Baseline 예측. Naive = 직전 달 값 반복, SeasonalNaive = 작년 같은 달 값, AutoARIMA = 자동 차수 선택 ARIMA.
    예측 구간(level 20/40/60/80)을 분위수 0.1~0.9로 변환해 FM과 같은 형식으로 저장.
    HistQuantile(과거 분포)도 함께 생성."""
    run_hist_quantile(data, out_dir)
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, Naive, SeasonalNaive

    m_ = data.season
    zoo = {"Naive": Naive(), "AutoARIMA": AutoARIMA(season_length=m_)}
    if m_ > 1:
        zoo["SeasonalNaive"] = SeasonalNaive(season_length=m_)   # 작년 같은 달 값
    if models is None:
        models = tuple(zoo)
    df = pd.DataFrame({
        "unique_id": np.repeat(data.ids, [len(c) for c in data.contexts]),
        "ds": np.concatenate([np.arange(len(c)) for c in data.contexts]),   # 정수 시간축
        "y": np.concatenate(data.contexts).astype(np.float64),
    })
    import time
    paths = []
    for m in models:
        t0 = time.time()
        sf = StatsForecast(models=[zoo[m]], freq=1, n_jobs=n_jobs)
        fc = sf.forecast(df=df, h=data.horizon, level=[20, 40, 60, 80]).sort_values(["unique_id", "ds"])
        elapsed = time.time() - t0
        cols = [f"{m}-lo-80", f"{m}-lo-60", f"{m}-lo-40", f"{m}-lo-20", m,
                f"{m}-hi-20", f"{m}-hi-40", f"{m}-hi-60", f"{m}-hi-80"]
        q = fc.set_index("unique_id").loc[data.ids, cols].to_numpy()
        q = np.nan_to_num(q, nan=0.0).reshape(len(data), data.horizon, len(QUANTILES))
        paths.append(save_predictions(data, q, m, out_dir, elapsed_sec=elapsed, n_params="통계 (CPU)"))
    return paths


# ---------------------------------------------------------------- 지표
def _pred_files(pred_dir):
    """pred_<모델>.csv 또는 pred_<모델>.csv.gz (같은 모델이 둘 다 있으면 .csv 우선)."""
    fs = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "pred_*.csv.gz"))) + sorted(glob.glob(os.path.join(pred_dir, "pred_*.csv"))):
        fs[_pred_name(f)] = f
    return [fs[k] for k in sorted(fs)]


def _pred_name(f):
    b = os.path.basename(f)
    return b[5:-7] if b.endswith(".csv.gz") else b[5:-4]


def _pinball(y, qpred):
    """y: (n,h), qpred: (n,h,Q) -> 분위수별 pinball 합 (n,Q)"""
    diff = y[..., None] - qpred
    qs = np.asarray(QUANTILES)
    return np.maximum(qs * diff, (qs - 1) * diff).sum(axis=1)


def evaluate(data: EvalData, pred_dir: str, models: list[str] | None = None, clip_negative: bool = True):
    """예측 파일들을 읽어 시계열별 지표와 요약표를 만든다.

    - MAE  : 평균 |실제 - 중앙값 예측|
    - MASE : MAE / (train 구간 계절 naive 예측 오차 평균, 시차 m=data.season)
             train 구간이 일정해서 분모가 0인 시계열은 제외(NaN)
    - WQL  : 2 * Σ pinball / Σ|실제| 를 분위수 9개 평균.
             요약표에서는 시계열을 모두 합쳐서 계산 (실제값 합이 0인 시계열 문제 방지)
    """
    files = _pred_files(pred_dir)
    names = [_pred_name(f) for f in files]
    if models is not None:
        files = [f for f, n in zip(files, names) if n in models]
        names = [n for n in names if n in models]
    if not files:
        raise FileNotFoundError(f"{pred_dir}에 pred_*.csv 파일이 없음")

    y = np.stack(data.actuals)                          # (n,h)
    m = data.season
    scale = np.array([np.mean(np.abs(c[m:] - c[:-m])) for c in data.contexts], dtype=np.float64)
    scale[scale == 0] = np.nan
    abs_y = np.abs(y).sum(axis=1)                       # (n,)

    rows, missing = [], {}
    for name, f in zip(names, files):
        p = pd.read_csv(f, parse_dates=["ds"])
        p = p[p.unique_id.isin(data.ids)]
        if p.unique_id.nunique() != len(data):
            missing[name] = len(data) - p.unique_id.nunique()
            continue
        p = p.set_index(["unique_id", "ds"]).sort_index()
        q = p[QCOLS].to_numpy().reshape(len(data), data.horizon, len(QUANTILES))
        q = np.sort(q, axis=-1)                         # 분위수 교차 방지
        if clip_negative:
            q = np.clip(q, 0, None)                     # 지출은 음수가 될 수 없음
        point = q[..., QUANTILES.index(0.5)]
        mae = np.abs(y - point).mean(axis=1)
        pin = _pinball(y, q)                            # (n,Q)
        rows.append(pd.DataFrame({
            "model": name, "unique_id": data.ids,
            "MAE": mae, "MASE": mae / scale,
            "_pin_sum": pin.mean(axis=1) * 2,           # 분위수 평균 후 ×2 → 합쳐서 WQL 계산용
            "_abs_y": abs_y,
        }))
    if missing:
        print("[경고] 시계열이 빠진 예측 파일은 제외:", missing)
    per = pd.concat(rows, ignore_index=True).join(data.meta, on="unique_id")
    with np.errstate(invalid="ignore", divide="ignore"):
        per["WQL"] = per._pin_sum / per._abs_y

    def summarize(g):
        return pd.Series({
            "MAE": g.MAE.mean(),
            "MASE": g.MASE.mean(),
            "MASE(중앙값)": g.MASE.median(),
            "WQL": g._pin_sum.sum() / g._abs_y.sum(),
            "시계열 수": len(g),
            "MASE 제외 수": int(g.MASE.isna().sum()),
        })

    cats = per[per.category != "전체"]                  # 전체 = 카테고리 합계라 중복 집계 방지
    overall = cats.groupby("model").apply(summarize, include_groups=False)
    by_cat = per.groupby(["category", "model"]).apply(summarize, include_groups=False)
    per = per.drop(columns=["_pin_sum", "_abs_y"])
    return per, overall, by_cat


def evaluate_by_month(data: EvalData, pred_dir: str, models: list[str] | None = None, clip_negative: bool = True):
    """예측 기간의 달마다 MAE·MASE·WQL 계산.

    같은 달(ds)의 모든 시계열을 모아서 계산함:
    - MAE  : 그 달의 |실제 - 중앙값 예측| 평균 ($)
    - MASE : 그 달의 |오차| ÷ 시계열별 척도(train 구간 계절 naive 오차, m=data.season)의 평균
    - WQL  : 그 달의 2 × Σ pinball ÷ Σ|실제| (분위수 9개 평균)
    반환: monthly (model, ds, 지표) — '전체' 카테고리 제외 / monthly_cat (category, model, ds, 지표)
    """
    files = _pred_files(pred_dir)
    names = [_pred_name(f) for f in files]
    if models is not None:
        files = [f for f, n in zip(files, names) if n in models]
        names = [n for n in names if n in models]
    y = np.stack(data.actuals)                                    # (n,h)
    m = data.season
    scale = np.array([np.mean(np.abs(c[m:] - c[:-m])) for c in data.contexts], dtype=np.float64)
    scale[scale == 0] = np.nan
    ds = pd.to_datetime(data.future_ds[0])
    cats = data.meta.category.to_numpy()
    out = []
    for name, f in zip(names, files):
        p = pd.read_csv(f, parse_dates=["ds"])
        p = p[p.unique_id.isin(data.ids)]
        if p.unique_id.nunique() != len(data):
            print(f"[경고] {name}: 시계열이 빠져서 제외")
            continue
        p = p.set_index(["unique_id", "ds"]).sort_index()
        q = np.sort(p[QCOLS].to_numpy().reshape(len(data), data.horizon, len(QUANTILES)), axis=-1)
        if clip_negative:
            q = np.clip(q, 0, None)
        err = np.abs(y - q[..., QUANTILES.index(0.5)])            # (n,h)
        diff = y[..., None] - q
        qs = np.asarray(QUANTILES)
        pin = np.maximum(qs * diff, (qs - 1) * diff).mean(axis=-1) * 2   # (n,h)
        for t, d in enumerate(ds):
            out.append(pd.DataFrame({"model": name, "ds": d, "category": cats,
                                     "err": err[:, t], "serr": err[:, t] / scale,
                                     "pin": pin[:, t], "absy": np.abs(y[:, t])}))
    long = pd.concat(out, ignore_index=True)

    def agg(g):
        return pd.Series({"MAE": g.err.mean(), "MASE": g.serr.mean(),
                          "WQL": g.pin.sum() / g.absy.sum() if g.absy.sum() > 0 else np.nan})

    monthly = long[long.category != "전체"].groupby(["model", "ds"]).apply(agg, include_groups=False).reset_index()
    monthly_cat = long.groupby(["category", "model", "ds"]).apply(agg, include_groups=False).reset_index()
    for t in (monthly, monthly_cat):
        t["month"] = t.ds.dt.strftime("%Y-%m")
    return monthly, monthly_cat


def month_tables(monthly: pd.DataFrame):
    """월별 결과를 지표마다 (행 = 달, 열 = 모델) 표로 펼침."""
    return {k: monthly.pivot(index="month", columns="model", values=k) for k in ["MAE", "MASE", "WQL"]}


def report(per: pd.DataFrame, overall: pd.DataFrame, by_cat: pd.DataFrame, pred_dir: str,
           baselines=("Naive", "SeasonalNaive", "AutoARIMA", "HistQuantile")):
    """팀 설명용 요약표.
    - 종합표: 지표별 순위, 시계열별 1위 비율, Naive보다 나은 시계열 비율(MASE<1), 추론 시간
    - 카테고리별 1위 모델 (MASE, WQL)
    """
    cats = per[per.category != "전체"].dropna(subset=["MASE"])
    piv = cats.pivot(index="unique_id", columns="model", values="MASE")
    win = piv.idxmin(axis=1).value_counts(normalize=True).reindex(piv.columns).fillna(0) * 100
    beat_naive = (piv < 1).mean() * 100

    t = overall[["MAE", "MASE", "WQL"]].copy()
    t["MASE 순위"] = t.MASE.rank().astype(int)
    t["WQL 순위"] = t.WQL.rank().astype(int)
    t["시계열별 1위 비율(%)"] = win.round(1)
    t["MASE<1 비율(%)"] = beat_naive.round(1)
    rp = os.path.join(pred_dir, "runtime.csv")
    if os.path.exists(rp):
        t = t.join(pd.read_csv(rp).set_index("model")[["추론 시간(초)", "파라미터"]])
    t["구분"] = ["Baseline" if m in baselines else "Foundation" for m in t.index]
    t = t.sort_values(["MASE 순위", "WQL 순위"])

    best = pd.DataFrame({
        "MASE 1위": by_cat.MASE.unstack("model").idxmin(axis=1),
        "MASE 값": by_cat.MASE.unstack("model").min(axis=1).round(3),
        "WQL 1위": by_cat.WQL.unstack("model").idxmin(axis=1),
        "WQL 값": by_cat.WQL.unstack("model").min(axis=1).round(3),
    })
    return t, best
