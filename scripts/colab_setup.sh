#!/usr/bin/env bash
# Colab(T4 GPU) 환경 준비: (가공 데이터가 없을 때만) 원본 다운로드·전처리 → 패키지 설치
set -e
cd "$(dirname "$0")/.."
mkdir -p raw data
if [ ! -f data/tabformer_monthly_long.csv ] && [ ! -f data/tabformer_monthly_long.csv.gz ]; then
  wget -q -c -O raw/transactions.tgz https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz
  tar xzf raw/transactions.tgz -C raw
  python src/prep_tabformer_monthly.py raw/card_transaction.v1.csv data/
fi
pip install -q "timesfm[torch]==3.0.2" "chronos-forecasting>=2.2"
pip install -q --no-deps uni2ts==2.0.0
pip install -q gluonts jaxtyping lightning einops einx CoLT5-attention beartype utilsforecast absl-py
[ -d third_party/FinCast-fts ] || git clone -q https://github.com/vincent05r/FinCast-fts.git third_party/FinCast-fts
echo SETUP_DONE
