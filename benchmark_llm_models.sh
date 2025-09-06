#!/bin/bash

# LLM Model Benchmarking Script
# This script runs individual benchmarks for each model/quantization combination
# to avoid sequential loading issues in llama-cpp-python

set -e  # Exit on error

# Configuration
SAMPLE_RATIO=${1:-0.01}  # Default to 1% of dataset, or use first argument
RESULTS_FILE="benchmark_results.json"

echo "========================================="
echo "LLM Model Benchmark Suite"
echo "Sample ratio: $SAMPLE_RATIO ($(echo "$SAMPLE_RATIO * 100" | bc)% of dataset)"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Models and quantizations to test
declare -a MODELS=("Qwen3-14B" "gemma-3-12b-it" "gpt-oss-20b")
declare -a QUANTIZATIONS=("Q4_K_XL" "Q5_K_XL" "Q6_K_XL")

# Track results
echo "[" > "$RESULTS_FILE"
FIRST_RESULT=true

# Function to run a single benchmark
run_benchmark() {
    local model=$1
    local quant=$2
    
    echo "----------------------------------------"
    echo "Testing: $model with $quant"
    echo "----------------------------------------"
    
    # Run the benchmark and capture output
    if uv run python -m kaggle_map.reranker.llm \
        --model "$model" \
        --quantization "$quant" \
        --sample-ratio "$SAMPLE_RATIO" 2>&1 | tee "benchmark_${model}_${quant}.log"; then
        
        echo "✅ Success: $model with $quant"
        
        # Extract MAP@3 score from log (you may need to adjust this regex)
        MAP_SCORE=$(grep -oP 'MAP@3 = \K[0-9.]+' "benchmark_${model}_${quant}.log" | tail -1 || echo "0.0")
        
        # Add to results file
        if [ "$FIRST_RESULT" = false ]; then
            echo "," >> "$RESULTS_FILE"
        fi
        FIRST_RESULT=false
        
        echo "  {" >> "$RESULTS_FILE"
        echo "    \"model\": \"$model\"," >> "$RESULTS_FILE"
        echo "    \"quantization\": \"$quant\"," >> "$RESULTS_FILE"
        echo "    \"map_score\": $MAP_SCORE," >> "$RESULTS_FILE"
        echo "    \"timestamp\": \"$(date -Iseconds)\"" >> "$RESULTS_FILE"
        echo -n "  }" >> "$RESULTS_FILE"
        
    else
        echo "❌ Failed: $model with $quant"
        echo "   Check benchmark_${model}_${quant}.log for details"
    fi
    
    # Small delay between models to ensure clean GPU state
    sleep 2
}

# Test each combination
for model in "${MODELS[@]}"; do
    for quant in "${QUANTIZATIONS[@]}"; do
        # Check if this combination is supported
        # (You may want to add logic here to skip unsupported combinations)
        
        run_benchmark "$model" "$quant"
        
        echo ""
    done
done

# Close JSON array
echo "" >> "$RESULTS_FILE"
echo "]" >> "$RESULTS_FILE"

echo "========================================="
echo "Benchmark Complete!"
echo "End time: $(date)"
echo "Results saved to: $RESULTS_FILE"
echo "========================================="

# Display summary
echo ""
echo "Summary of results:"
echo "-------------------"
if command -v jq &> /dev/null; then
    jq -r '.[] | "\(.model) \(.quantization): MAP@3 = \(.map_score)"' "$RESULTS_FILE" | sort -t'=' -k2 -rn
else
    echo "Install jq for formatted output: sudo apt-get install jq"
    cat "$RESULTS_FILE"
fi