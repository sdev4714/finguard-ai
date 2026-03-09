#!/bin/bash
set -e

export PYTHONPATH=/opt/render/project/src

echo "=== Running ML Pipeline ==="

python data/generate_data.py
echo "✅ Data generated"

python src/preprocess.py
echo "✅ Preprocessing done"

python src/train.py
echo "✅ Training done"

python src/evaluate.py
echo "✅ Evaluation done"

python src/explain.py
echo "✅ Explainer built"

python src/fairness.py
echo "✅ Fairness audit done"

echo "=== Starting Server ==="
uvicorn app.main:app --host 0.0.0.0 --port $PORT