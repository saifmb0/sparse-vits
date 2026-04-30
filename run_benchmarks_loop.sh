#!/bin/bash

# Number of repeated iterations (override with first arg, e.g. ./run_benchmarks_loop.sh 50)
N=${1:-100}

echo "Starting benchmarks: N=$N iterations"
echo "Logs will be saved within each timestamped directory."

trap "echo 'Loop interrupted by user'; exit" INT

# ── One-time setup: correctness + profiler trace + extended benchmarks ────────
echo "========================================================"
echo "ONE-TIME: Correctness + profiler + model scaling + sparsity sweep + high-res + large-seqlen"
echo "========================================================"
python3 pipeline_analysis.py --mode correctness   --results-dir results/RTX4000Ada
python3 pipeline_analysis.py --mode profile       --results-dir results/RTX4000Ada
python3 extended_e2e.py      --mode model-scaling  --results-dir results/RTX4000Ada
python3 extended_e2e.py      --mode sparsity-sweep --results-dir results/RTX4000Ada
python3 extended_e2e.py      --mode high-res       --results-dir results/RTX4000Ada
python3 micro_benchmark.py   --mode large-seqlen   --results-dir results/RTX4000Ada --iters 200

# ── Repeated loop ─────────────────────────────────────────────────────────────
for ((i=1; i<=N; i++)); do
    TIMESTAMP=$(date +%Y%m%d%H%M%S)

    echo "========================================================"
    echo "ITERATION $i / $N : $TIMESTAMP"
    echo "========================================================"

    # 1. E2E Benchmark
    E2E_DIR="results/e2e_benchmark/$TIMESTAMP"
    mkdir -p "$E2E_DIR"
    echo "Running E2E..."
    python3 e2e_benchmark.py --results-dir "$E2E_DIR"

    # 2. Micro Benchmark
    MICRO_DIR="results/micro_benchmark/$TIMESTAMP"
    mkdir -p "$MICRO_DIR"
    echo "Running Microbench..."
    python3 micro_benchmark.py --results-dir "$MICRO_DIR" --iters 300

    # 3. Stage Breakdown
    STAGE_DIR="results/stage_breakdown/$TIMESTAMP"
    mkdir -p "$STAGE_DIR"
    echo "Running Stage Breakdown..."
    python3 pipeline_analysis.py --mode stage-breakdown --results-dir "$STAGE_DIR"

    # 4. Validate outputs
    if [ ! -f "$E2E_DIR/e2e_benchmark.json" ] || \
       [ ! -f "$MICRO_DIR/micro_benchmark.json" ] || \
       [ ! -f "$STAGE_DIR/stage_breakdown.json" ]; then
        echo "WARNING: One or more benchmarks did not produce JSON."
    else
        echo "Iteration $i / $N complete."
    fi

    if [ $i -lt $N ]; then
        echo "Cooldown sleep (15s)..."
        sleep 15
    fi
done

echo "Benchmark loop finished."
echo "Running final aggregation..."
python3 aggregate_results.py
