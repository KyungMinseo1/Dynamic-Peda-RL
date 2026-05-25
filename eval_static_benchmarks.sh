#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}❌ Error: $1 is not installed${NC}"
        exit 1
    fi
}

echo -e "${YELLOW}[*] Checking dependencies...${NC}"
check_dependency "lm_eval"
check_dependency "python"
echo -e "${GREEN}[✓] All dependencies found${NC}\n"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODELS=(
    "models/Dynamic-RL-7b_dynamic_w_uniformshareacc"
)

BENCHMARK_DIR="logs/benchmarks/${TIMESTAMP}"
mkdir -p "$BENCHMARK_DIR"

echo -e "${YELLOW}[*] Benchmark results will be saved to: $BENCHMARK_DIR${NC}\n"

TOTAL_MODELS=${#MODELS[@]}
COMPLETED_MODELS=0
FAILED_MODELS=0

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | sed 's/\//_/g')
    MODEL_LOG_DIR="${BENCHMARK_DIR}/${MODEL_NAME}"
    mkdir -p "$MODEL_LOG_DIR"
    
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}Evaluating Model: $MODEL${NC}"
    echo -e "${YELLOW}============================================================${NC}\n"
    
    # 1. MMLU (5-shot)
    echo -e "${YELLOW}[1/3] Evaluating MMLU (5-shot)...${NC}"
    if lm_eval --model vllm \
        --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8 \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --apply_chat_template \
        --output_path "$MODEL_LOG_DIR"/mmlu 2>&1 | tee "$MODEL_LOG_DIR"/mmlu.log; then
        echo -e "${GREEN}[✓] MMLU completed${NC}\n"
    else
        echo -e "${RED}[✗] MMLU failed${NC}\n"
        FAILED_MODELS=$((FAILED_MODELS + 1))
        continue
    fi

    # 2. GSM8K (4-shot)
    echo -e "${YELLOW}[2/3] Evaluating GSM8K (4-shot)...${NC}"
    if lm_eval --model vllm \
        --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,max_model_len=8192,enable_prefix_caching=True \
        --tasks gsm8k \
        --num_fewshot 4 \
        --batch_size auto \
        --apply_chat_template \
        --system_instruction "Please reason step by step, and put your final answer after ####." \
        --gen_kwargs max_gen_toks=2048 \
        --output_path "$MODEL_LOG_DIR"/gsm8k 2>&1 | tee "$MODEL_LOG_DIR"/gsm8k.log; then
        echo -e "${GREEN}[✓] GSM8K completed${NC}\n"
    else
        echo -e "${RED}[✗] GSM8K failed${NC}\n"
        FAILED_MODELS=$((FAILED_MODELS + 1))
        continue
    fi

    # 3. MATH500 (0-shot)
    echo -e "${YELLOW}[3/3] Evaluating MATH500 (0-shot)...${NC}"
    if lm_eval --model vllm \
        --model_args pretrained=$MODEL,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,max_model_len=8192,enable_prefix_caching=True \
        --tasks minerva_math500 \
        --num_fewshot 0 \
        --batch_size auto \
        --apply_chat_template \
        --system_instruction "Solve the problem carefully step by step and put your final answer within \boxed{}." \
        --gen_kwargs max_gen_toks=4096 \
        --output_path "$MODEL_LOG_DIR"/math500 2>&1 | tee "$MODEL_LOG_DIR"/math500.log; then
        echo -e "${GREEN}[✓] MATH500 completed${NC}\n"
    else
        echo -e "${RED}[✗] MATH500 failed${NC}\n"
        FAILED_MODELS=$((FAILED_MODELS + 1))
        continue
    fi

    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
    echo -e "${GREEN}[✓] Done evaluating $MODEL${NC}\n"
done

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Benchmark Summary${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo -e "Total Models: $TOTAL_MODELS"
echo -e "${GREEN}Completed: $COMPLETED_MODELS${NC}"
echo -e "${RED}Failed: $FAILED_MODELS${NC}"
echo -e "Results saved to: $BENCHMARK_DIR"
echo -e "${YELLOW}============================================================${NC}\n"

if [ $FAILED_MODELS -eq 0 ]; then
    echo -e "${GREEN}[✓] All evaluations completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}[✗] Some evaluations failed. Check logs for details.${NC}"
    exit 1
fi
