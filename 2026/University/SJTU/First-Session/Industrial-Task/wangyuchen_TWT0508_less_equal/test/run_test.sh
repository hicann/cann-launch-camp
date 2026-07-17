#!/bin/bash
# Run local numpy verification (no NPU required)
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  LessEqual Operator Local Test"
echo "  dtypes: float16 / float32 / int32 / int8"
echo "  coverage: 1D/2D/3D/4D, broadcast, extreme values, large arrays"
echo "=========================================="
echo ""

python3 gen_data.py

echo ""
echo "=== local verification done ==="
echo "Next step: submit .cpp/.h files to CANNJudge"
