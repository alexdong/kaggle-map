#!/bin/bash
# Run the LLM benchmark overnight with error recovery

# Set working directory to script location
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file for this run
LOGFILE="logs/overnight_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================" | tee -a "$LOGFILE"
echo "Starting LLM Benchmark - $(date)" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Configuration:" | tee -a "$LOGFILE"
echo "- Sample ratio: 100% (full validation set)" | tee -a "$LOGFILE"
echo "- Models: GPT-OSS 20B, GEMMA 3 27B" | tee -a "$LOGFILE"
echo "- Quantizations: 6 total combinations" | tee -a "$LOGFILE"
echo "- Data: datasets/33474_focus_group.csv" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Run the benchmark with full data (100% sample)
# Use nohup to prevent termination on disconnect
nohup uv run python benchmark_llm_models.py \
    --data-path datasets/33474_focus_group.csv \
    --sample-ratio 1.0 \
    >> "$LOGFILE" 2>&1 &

# Get the process ID
PID=$!
echo "Benchmark started with PID: $PID" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "To monitor progress:" | tee -a "$LOGFILE"
echo "  tail -f $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "To check if still running:" | tee -a "$LOGFILE"
echo "  ps -p $PID" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Results will be saved to: logs/llm_benchmark_*.csv" | tee -a "$LOGFILE"