#!/bin/bash
set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install numpy==1.26.4
pip install -r requirements.txt

echo "=== Running ML Pipeline ==="
export PYTHONPATH=/opt/render/project/src

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

echo "=== Checking models directory ==="
ls -la models/

echo "=== Build complete ==="