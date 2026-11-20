#!/usr/bin/env bash
# Re-train Lily's PullNet 3-hop model with the code's default settings, then score the test set
# with her own evaluate_inference.py. Run from anywhere; everything is written under analysis/pullnet_run/.
set -u
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jjding/9444/.venv-analysis/bin/python
cd "$RUN_DIR"
mkdir -p logs results
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done; }

{
  echo "run started: $(date '+%F %T')"
  echo "code: commit $(git -C ../.. rev-parse --short HEAD) + local fix (pullnet/graph.py: fact_tensors() uncommented)"
  echo "run via run_with_gpu_cleanup.py (frees unused GPU memory after each chunk; no change to the model or maths)"
  "$PY" -c "import torch, platform; print('python', platform.python_version(), '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
  echo "command: python pullnet/train.py --hop 3   (all other settings = code defaults)"
} | tee logs/train_hop3.log

/usr/bin/time -v -o logs/train_hop3.time timeout 14400 \
  "$PY" -u run_with_gpu_cleanup.py ../../pullnet/train.py --hop 3 2>&1 | stamp | tee -a logs/train_hop3.log
grep -E "Elapsed \(wall|Maximum resident|Exit status" logs/train_hop3.time | tee -a logs/train_hop3.log

echo "test scoring started: $(date '+%F %T')" | tee logs/eval_hop3_test.log
/usr/bin/time -v -o logs/eval_hop3_test.time \
  "$PY" -u ../../pullnet/evaluate_inference.py --hop 3 --split test --device cpu \
  --output results/hop3_test_full.json 2>&1 | grep -vE "^ +[0-9]+/[0-9]+ questions$" | stamp | tee -a logs/eval_hop3_test.log
grep -E "Elapsed \(wall|Maximum resident|Exit status" logs/eval_hop3_test.time | tee -a logs/eval_hop3_test.log

echo "saving every answer: $(date '+%F %T')" | tee logs/predict_hop3_test.log
"$PY" -u ../pullnet_predict.py --hop 3 --split test 2>&1 | stamp | tee -a logs/predict_hop3_test.log
