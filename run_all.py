#!/usr/bin/env python3
"""
Master runner — executes all three benchmarks and generates plots.
Usage:
    python run_all.py               # run all benchmarks
    python run_all.py --bench 1     # run only benchmark 1
    python run_all.py --bench 2     # run only benchmark 2
    python run_all.py --bench 3     # run only benchmark 3
    python run_all.py --plot        # re-plot from saved JSON results
"""

import sys, os, argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def verify_environment():
    print("=" * 60)
    print("Ragged-Batch ViT Inference Engine — Environment Check")
    print("=" * 60)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    import triton
    print(f"  Triton  : {triton.__version__}")
    import timm
    print(f"  timm    : {timm.__version__}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=int, default=0, help="Run specific benchmark (1/2/3), 0=all")
    parser.add_argument("--plot", action="store_true", help="Only re-generate plots from saved results")
    args = parser.parse_args()

    verify_environment()

    if args.plot:
        print("\n>>> Re-generating plots from saved results...\n")
        from benchmarks.bench1_throughput import plot_benchmark_1
        from benchmarks.bench2_sparsity  import plot_benchmark_2
        from benchmarks.bench3_vram      import plot_benchmark_3
        from benchmarks.bench4_model_scaling import plot_benchmark_4
        from benchmarks.bench5_accuracy  import plot_benchmark_5
        from benchmarks.bench6_sota_baselines import plot_benchmark_6
        if args.bench in (0, 1): plot_benchmark_1()
        if args.bench in (0, 2): plot_benchmark_2()
        if args.bench in (0, 3): plot_benchmark_3()
        if args.bench in (0, 4): plot_benchmark_4()
        if args.bench in (0, 5): plot_benchmark_5()
        if args.bench in (0, 6): plot_benchmark_6()
        return

    os.makedirs("results", exist_ok=True)

    if args.bench in (0, 1):
        print("\n" + "=" * 60)
        print("BENCHMARK 1: Batch-Size Scaling (Throughput)")
        print("=" * 60)
        from benchmarks.bench1_throughput import run_benchmark_1
        run_benchmark_1()

    torch.cuda.empty_cache()

    if args.bench in (0, 2):
        print("\n" + "=" * 60)
        print("BENCHMARK 2: Sparsity vs. Speedup")
        print("=" * 60)
        from benchmarks.bench2_sparsity import run_benchmark_2
        run_benchmark_2()

    torch.cuda.empty_cache()

    # if args.bench in (0, 3):
    #     print("\n" + "=" * 60)
    #     print("BENCHMARK 3: Peak VRAM Allocation")
    #     print("=" * 60)
    #     from benchmarks.bench3_vram import run_benchmark_3
    #     run_benchmark_3()

    torch.cuda.empty_cache()

    if args.bench in (0, 4):
        print("\n" + "=" * 60)
        print("BENCHMARK 4: Model-Size Scaling (Tiny/Small/Base)")
        print("=" * 60)
        from benchmarks.bench4_model_scaling import run_benchmark_4
        run_benchmark_4()

    torch.cuda.empty_cache()

    if args.bench in (0, 5):
        print("\n" + "=" * 60)
        print("BENCHMARK 5: Accuracy vs. Efficiency (Pareto Frontier)")
        print("=" * 60)
        from benchmarks.bench5_accuracy import run_benchmark_5
        run_benchmark_5()

    torch.cuda.empty_cache()

    if args.bench in (0, 6):
        print("\n" + "=" * 60)
        print("BENCHMARK 6: SOTA Systems Baselines")
        print("=" * 60)
        from benchmarks.bench6_sota_baselines import run_benchmark_6
        run_benchmark_6()

    print("\n" + "=" * 60)
    print("All benchmarks complete. Results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
