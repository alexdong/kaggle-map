#!/bin/bash

# Compare different quantization levels for LLM models
# Note: This requires a GPU to run efficiently

echo "========================================="
echo "LLM Quantization Comparison"
echo "Model: gemma-3-12b-it"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Run benchmarks for each quantization level
# Using 10% sample ratio for faster testing (adjust as needed)
SAMPLE_RATIO=1

echo "Testing Q2_K_XL (smallest, fastest, lowest quality)..."
uv run -m kaggle_map.reranker.llm \
    --model gemma-3-12b-it \
    --quantization Q2_K_XL \
    --sample-ratio $SAMPLE_RATIO > results_q2.txt 2>&1

echo "Testing Q3_K_XL (balanced size/quality)..."
uv run -m kaggle_map.reranker.llm \
    --model gemma-3-12b-it \
    --quantization Q3_K_XL \
    --sample-ratio $SAMPLE_RATIO > results_q3.txt 2>&1

echo "Testing Q4_K_XL (good quality/speed balance)..."
uv run -m kaggle_map.reranker.llm \
    --model gemma-3-12b-it \
    --quantization Q4_K_XL \
    --sample-ratio $SAMPLE_RATIO > results_q4.txt 2>&1

echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="
echo ""

# Extract MAP@3 scores from results
echo "Q2_K_XL MAP@3: $(grep -o 'MAP@3.*[0-9]\.[0-9]*' results_q2.txt | tail -1)"
echo "Q3_K_XL MAP@3: $(grep -o 'MAP@3.*[0-9]\.[0-9]*' results_q3.txt | tail -1)"
echo "Q4_K_XL MAP@3: $(grep -o 'MAP@3.*[0-9]\.[0-9]*' results_q4.txt | tail -1)"

echo ""
echo "Detailed results saved in:"
echo "  - results_q2.txt"
echo "  - results_q3.txt"
echo "  - results_q4.txt"
echo ""
echo "End time: $(date)"
echo "========================================="