#include <immintrin.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <string>

volatile size_t global_sink = 0; //to force the compiler to actually run the functions and not just cut them out
//to be compiled with icpx using qopenmp and march=native flags


size_t simd_char_find(const char* haystack, size_t length, char target) {
    __m256i target_reg = _mm256_set1_epi8(target);
    size_t i = 0;
    for (; i + 128 <= length; i += 128) {
        _mm_prefetch((char*)(haystack + i + 1024), _MM_HINT_T0);
        __m256i c1 = _mm256_load_si256((const __m256i*)(haystack + i));
        __m256i c2 = _mm256_load_si256((const __m256i*)(haystack + i + 32));
        __m256i c3 = _mm256_load_si256((const __m256i*)(haystack + i + 64));
        __m256i c4 = _mm256_load_si256((const __m256i*)(haystack + i + 96));

        __m256i cmp = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(c1, target_reg), 
                                                      _mm256_cmpeq_epi8(c2, target_reg)),
                                      _mm256_or_si256(_mm256_cmpeq_epi8(c3, target_reg), 
                                                      _mm256_cmpeq_epi8(c4, target_reg)));

        if (!_mm256_testz_si256(cmp, cmp)) {
            int m;
            if ((m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, target_reg)))) return i + _tzcnt_u32(m);
            if ((m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, target_reg)))) return i + 32 + _tzcnt_u32(m);
            if ((m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(c3, target_reg)))) return i + 64 + _tzcnt_u32(m);
            return i + 96 + _tzcnt_u32(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c4, target_reg)));
        }
    }
    for (; i < length; ++i) { if (haystack[i] == target) return i; }
    return (size_t)-1;
}


size_t find_char_parallel_simd(const char* haystack, size_t length, char target) {
    size_t final_res = -1;

    #pragma omp parallel
    {
        size_t local_res = -1;
        #pragma omp for nowait
        for (size_t i = 0; i < length; i += 1024*1024*2) { 
            if (local_res != -1) continue; 
            size_t cur_len = std::min((size_t)1024*1024*2, length - i);
            size_t res = simd_char_find(haystack + i, cur_len, target);
            if (res != -1) local_res = i + res;
        }
        #pragma omp critical
        {
            if (local_res != -1 && (final_res == -1 || local_res < final_res)) 
                final_res = local_res;
        }
    }
    return final_res;
}



struct Stats {
    std::string name;
    double mean;
    double stddev;
    double throughput;
};


void head_to_head(const Stats& a, const Stats& b) {
    double diff = std::abs(a.mean - b.mean);
    double threshold = 2.0 * (a.stddev + b.stddev); 
    
    std::cout << "Matchup: [" << a.name << "] vs [" << b.name << "]" << std::endl;
    std::cout << "  Delta: " << std::fixed << std::setprecision(4) << diff << " ms";

    if (diff <= threshold) {
        std::cout << " -> RESULT: DRAW (Statistical Tie)" << std::endl;
        std::cout << "  Verdict: Difference is within the margin of noise." << std::endl;
    } else {
        const Stats& winner = (a.mean < b.mean) ? a : b;
        double speedup = (std::max(a.mean, b.mean) / std::min(a.mean, b.mean));
        std::cout << " -> RESULT: CLEAR WINNER [" << winner.name << "]" << std::endl;
        std::cout << "  Verdict: " << winner.name << " is " << speedup << "x faster." << std::endl;
    }
    std::cout << std::string(60, '-') << std::endl;
}

void print_ranking(std::vector<Stats> results) {
    std::sort(results.begin(), results.end(), [](const Stats& a, const Stats& b) {
        return a.mean < b.mean;
    });

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                FINAL RANKING" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << i + 1 << ". " << std::left << std::setw(15) << results[i].name 
                  << " | " << std::fixed << std::setprecision(2) << results[i].mean << " ms"
                  << " | " << results[i].throughput << " GB/s" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
}

template <typename TFunc>
Stats run_benchmark(const std::string& name, int runs, size_t size, TFunc func) {
    std::vector<double> times;
    times.reserve(runs);
    global_sink = func(); 

    for (int i = 0; i < runs; ++i) {
        auto s = std::chrono::high_resolution_clock::now();
        
        size_t result = func();
        

        global_sink = result; 
        
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(e - s).count());
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / runs;
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stddev = std::sqrt(std::max(0.0, sq_sum / runs - mean * mean));
    double tp = (static_cast<double>(size) / (1024*1024*1024)) / (mean / 1000.0);

    return {name, mean, stddev, tp};
}

int main() {
    const size_t TOTAL_SIZE = 4ULL * 1024 * 1024 * 1024;
    const int RUNS = 100;
    std::vector<char> data(TOTAL_SIZE, 'B');
    data[TOTAL_SIZE - 500] = 'A';
    const char* ptr = data.data();

    std::cout << "Starting 100 run Statistical Analysis\n" << std::endl;

    auto b_naive    = run_benchmark("Naive Loop",  RUNS, TOTAL_SIZE, [&](){ for(size_t i=0; i<TOTAL_SIZE; ++i) if(ptr[i]=='A') return i; return (size_t)-1; });
    auto b_std      = run_benchmark("std::find",   RUNS, TOTAL_SIZE, [&](){ return (size_t)std::distance(data.begin(), std::find(data.begin(), data.end(), 'A')); });
    auto b_manual   = run_benchmark("AVX2 Manual", RUNS, TOTAL_SIZE, [&](){ return simd_char_find(ptr, TOTAL_SIZE, 'A'); });
    auto b_parallel = run_benchmark("OpenMP+AVX2", RUNS, TOTAL_SIZE, [&](){ return find_char_parallel_simd(ptr, TOTAL_SIZE, 'A'); });

    std::cout << "=== INDIVIDUAL MATCHUPS (Vs. AVX2 Manual) ===\n" << std::endl;
    head_to_head(b_manual, b_std);
    head_to_head(b_manual, b_naive);
    head_to_head(b_manual, b_parallel);

    print_ranking({b_naive, b_std, b_manual, b_parallel});

    return 0;
}