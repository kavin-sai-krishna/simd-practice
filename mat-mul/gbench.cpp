
#include "benchmark/benchmark.h"
#include "mat-mul.h"
#include <memory>

std::unique_ptr<Matrix> a = []() {
    auto mat = std::make_unique<Matrix>();
    init(*mat);
    return mat;
}();

std::unique_ptr<Matrix> b = []() {
    auto mat = std::make_unique<Matrix>();
    init(*mat);
    return mat;
}();

std::unique_ptr<Matrix> c (new Matrix());

static void bench_naive_for(benchmark::State &state) {

  for (auto _ : state) {
    naive_for(*c,*a,*b);
    benchmark::DoNotOptimize(c);
  }
}

static void bench_optimized_for(benchmark::State &state) {
  for (auto _ : state) {
    optimized_for(*c,*a,*b);
    benchmark::DoNotOptimize(c);
  }
}

static void bench_transpose_mul(benchmark::State &state) {
  for (auto _ : state) {
    transpose_mul(*c,*a,*b);
    benchmark::DoNotOptimize(c);
  }
}

static void bench_avx_multiply(benchmark::State &state) {
  for (auto _ : state) {
    avx_multiply(*c,*a,*b);
    benchmark::DoNotOptimize(c);
  }
}

// Register the function as a benchmark
BENCHMARK(bench_naive_for)->Iterations(ITERS)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_optimized_for)->Iterations(ITERS)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_transpose_mul)->Iterations(ITERS)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_avx_multiply)->Iterations(ITERS)->Unit(benchmark::kMillisecond);


// Run the benchmark
BENCHMARK_MAIN();