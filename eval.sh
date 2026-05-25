#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "==== Job Info ===="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=================="

START_TIME=$(date +%s)

echo "======================================"
echo "🚀 Eval Start : $(date)"
echo "======================================"

clear_gpu() {
    echo ""
    echo "🧹 Cleaning up lingering processes..."
    
    pkill -9 -f "VLLM" || true
    pkill -9 -f "vllm" || true
    sleep 5

    echo "💧 Clearing internal Torch cache..."
    python - <<'PY'
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Internal cache cleared")
PY

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "Current GPU memory usage:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    fi
    echo "--------------------------------------"
}

run_eval () {
    local name=$1
    local config=$2

    echo ""
    echo "▶️ Start $name : $(date)"
    python eval.py --config-name "$config"
    echo "✅ Done  $name : $(date)"
    clear_gpu
}

run_eval_original () {
    local name=$1
    local config=$2

    echo ""
    echo "▶️ Start $name : $(date)"
    python eval_original.py --config-name "$config"
    echo "✅ Done  $name : $(date)"
    clear_gpu
}

run_eval_integrated () {
    local name=$1
    local config=$2
    echo ""
    echo "▶️ Start $name (integrated) : $(date)"
    python eval_integrated.py --config-name "$config" $ARGS
    echo "✅ Done  $name (integrated) : $(date)"
    clear_gpu
}

run_eval_integrated_14B () {
    local name=$1
    local config=$2
    echo ""
    echo "▶️ Start $name (integrated_14B) : $(date)"
    python eval_integrated_14B.py --config-name "$config" $ARGS
    echo "✅ Done  $name (integrated_14B) : $(date)"
    clear_gpu
}

#====== 14B Student Eval (for main paper) ======
# run_eval_integrated_14B "TutorRL-7B_2000" "TutorRL-7b_2000.yaml"
# run_eval_integrated_14B "Qwen2.5-7B-Instruct" "Qwen-7b.yaml"
# run_eval_integrated_14B "DynamicRL-7B" "DynamicRL-7b_2000.yaml"
# run_eval_integrated_14B "DynamicRL-7B_4000" "DynamicRL-7b_4000.yaml"
# run_eval_integrated_14B "DynamicRL-7B_w_1.0_2000" "DynamicRL-7b_w_1.0_2000.yaml"
# run_eval_integrated_14B "Qwen-SFT" "Qwen-SFT.yaml"
# run_eval_integrated_14B "Qwen-MDPO_2000" "Qwen-MDPO.yaml"


# ====== 8B Student Eval (for main paper) ======
# run_eval_integrated "TutorRL-7B_2000" "TutorRL-7b_2000.yaml"
# run_eval_integrated "Qwen2.5-7B-Instruct" "Qwen-7b.yaml"
# run_eval_integrated "DynamicRL-7B" "DynamicRL-7b_2000.yaml"
# run_eval_integrated "DynamicRL-7B_4000" "DynamicRL-7b_4000.yaml"
# run_eval_integrated "DynamicRL-7B_w_1.0_2000" "DynamicRL-7b_w_1.0_2000.yaml"
# run_eval_integrated "Qwen-SFT" "Qwen-SFT.yaml"
# run_eval_integrated "Qwen-MDPO_2000" "Qwen-MDPO.yaml"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================"
echo "🎉 All Eval Finished : $(date)"
echo "⏱️ Total Time : $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
echo "======================================"
