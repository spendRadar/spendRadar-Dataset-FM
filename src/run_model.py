"""시계열 파운데이션 모델 4종 zero-shot 예측 → results/predictions/pred_<모델>.csv
GPU 권장 (Colab T4에서 검증). 사용법: python src/run_model.py {timesfm|moirai|chronos2|fincast} [--data ...]
"""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, torch
import spendradar_eval as se

ap = argparse.ArgumentParser(description='시계열 파운데이션 모델 zero-shot 예측')
ap.add_argument('model', choices=['timesfm', 'moirai', 'chronos2', 'fincast'])
ap.add_argument('--data', default=str(ROOT / 'data' / ('tabformer_monthly_long.csv' if (ROOT / 'data' / 'tabformer_monthly_long.csv').exists() else 'tabformer_monthly_long.csv.gz')))
ap.add_argument('--out', default=str(ROOT / 'results' / 'predictions'))
ap.add_argument('--fincast-src', default=str(ROOT / 'third_party' / 'FinCast-fts' / 'src'))
args = ap.parse_args()
MODEL = args.model
data = se.load_data(args.data)
print(data.describe(), flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev, flush=True)
OUT = args.out
H = data.horizon
if MODEL == 'timesfm':
    import timesfm
    torch.set_float32_matmul_precision('high')
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    m.compile(timesfm.ForecastConfig(max_context=128, max_horizon=128, normalize_inputs=True,
              use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
              fix_quantile_crossing=True, per_core_batch_size=64))
    t = time.time()
    _, quant = m.forecast(horizon=H, inputs=[c.astype(np.float64) for c in data.contexts])
    q = np.asarray(quant)[:, :H, 1:10]
    se.save_predictions(data, q, 'TimesFM2.5', OUT, elapsed_sec=time.time() - t, n_params='200M')
elif MODEL == 'moirai':
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    m = Moirai2Forecast(module=Moirai2Module.from_pretrained('Salesforce/moirai-2.0-R-small'),
                        prediction_length=H, context_length=max(len(c) for c in data.contexts),
                        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(dev)
    assert list(m.module.quantile_levels) == se.QUANTILES, m.module.quantile_levels
    t = time.time(); preds = []
    for b in se.batched(data.contexts, 256):
        o = np.asarray(m.predict(past_target=[c.astype(np.float32) for c in b]))
        if o.ndim == 4: o = o[..., 0]
        preds.append(o.transpose(0, 2, 1))
    se.save_predictions(data, np.concatenate(preds), 'Moirai2.0', OUT, elapsed_sec=time.time() - t, n_params='11M')
elif MODEL == 'chronos2':
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=dev)
    t = time.time(); preds = []
    for b in se.batched(data.contexts, 256):
        with torch.no_grad():
            qs, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in b], prediction_length=H,
                                           quantile_levels=se.QUANTILES)
        preds += [np.asarray(x.float().cpu()).reshape(H, 9) for x in qs]
    se.save_predictions(data, np.stack(preds), 'Chronos2', OUT, elapsed_sec=time.time() - t, n_params='120M')
elif MODEL == 'fincast':
    sys.path.insert(0, args.fincast_src)
    from types import SimpleNamespace
    from huggingface_hub import hf_hub_download
    from tools.inference_utils import get_model_api
    ck = hf_hub_download('Vincent05R/FinCast', 'v1.pth')
    api = get_model_api(SimpleNamespace(backend='gpu' if dev == 'cuda' else 'cpu', model_path=ck, horizon_len=H,
                        context_len=128, num_experts=4, gating_top_n=2, load_from_compile=True, forecast_mode='median'))
    t = time.time(); preds = []
    for b in se.batched(data.contexts, 64):
        _, full = api.forecast(list(b), freq=[1] * len(b))
        preds.append(np.asarray(full)[:, :H, 1:])
    se.save_predictions(data, np.concatenate(preds), 'FinCast', OUT, elapsed_sec=time.time() - t, n_params='1B')
print('OK', MODEL, flush=True)
