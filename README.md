# LunarSearch: High-Performance SIMD String Scanning

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LunarSearch** is a specialized string-searching utility designed to push the boundaries of the **Intel Lunar Lake** architecture. By utilizing hand-tuned AVX2 intrinsics and OpenMP parallelism, this project achieves near-theoretical peak memory bandwidth, successfully hitting the "Memory Wall" of modern mobile silicon.

## 🚀 Performance Overview
Benchmarks conducted on 4GB of contiguous memory show that **LunarSearch** effectively saturates the memory controller, doubling the performance of single-threaded standard library functions.

| Method | Mean Time (4GB) | Throughput | Status |
| :--- | :--- | :--- | :--- |
| **OpenMP + AVX2** | **50.77 ms** | **78.78 GB/s** | 🏆 **Champion** |
| **std::find (icpx)** | 111.29 ms | 35.94 GB/s | Statistical Tie |
| **Manual AVX2** | 112.15 ms | 35.67 GB/s | Statistical Tie |
| **Naive Loop (-O3)**| 138.88 ms | 28.80 GB/s | 1.3x Slower |

## 🛠️ The Challenge: Beating the Compiler
Standard auto-vectorization often fails to account for the specific prefetching needs and instruction-level parallelism (ILP) required to saturate high-speed LPDDR5x memory. Coming from a 10-month background in Python, this project was a deep-dive into "Mechanical Sympathy"—writing code that respects the physical constraints of the CPU and RAM.

### Key Optimizations:
* **Manual AVX2 Unrolling:** Bypasses compiler heuristics to process **128 bytes per iteration** using four 256-bit YMM registers.
* **Branchless Search logic:** Uses bitmask-to-index conversion via `_mm256_movemask_epi8` and `_tzcnt_u32` to keep the execution pipeline fluid.
* **Software Prefetching:** Utilizes `_mm_prefetch` to mask memory latency, ensuring data is in the L1/L2 cache before the CPU execution ports require it.
* **OpenMP Orchestration:** Parallelizes the SIMD core to open multiple concurrent pipelines to the memory controller, achieving a **2.2x scaling factor** over the single-core limit.

## 📊 Statistical Validation
Performance in systems programming is a science, not a feeling. This repository includes a high-precision benchmarking harness that:
1.  Performs **100-sample runs** for every implementation.
2.  Calculates **Mean** and **Standard Deviation** to filter out OS jitter.
3.  Applies a **2-Sigma Significance Test** to ensure that "Wins" are architecturally real and not just statistical noise.
![Performance Ranking](img/Screenshot_20260421_011415.png)

## 📦 Building & Running
This project is optimized for the **Intel oneAPI DPC++/C++ Compiler (icpx)**.

### Prerequisites
* Intel oneAPI Base Toolkit
* A CPU supporting AVX2 (Optimized for Intel Lunar Lake / Lion Cove)
* OpenMP 4.5+

### Compilation
```bash
icpx -qopenmp -O3 -march=native string_match_bench.cpp -o string_match_bench
```
