#!/bin/bash
set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install numpy==1.26.4
pip install -r requirements.txt

echo "=== Build complete ==="