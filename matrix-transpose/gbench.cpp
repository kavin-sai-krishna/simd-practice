
#include "benchmark/benchmark.h"
#include "matrix-transpose.h"
#include <memory>

std::unique_ptr<Matrix> a = []() {
    auto mat = std::make_unique<Matrix>();
    init(*mat);
    return mat;
}();

static void bench_tri(benchmark::State &state) {

  for (auto _ : state) {
    transpose_triangle(*a);
    benchmark::DoNotOptimize(a);
  }
}

static void bench_block(benchmark::State &state) {

  std::unique_ptr<Matrix> b(new Matrix());


  for (auto _ : state) {
    transpose_block(*a);
    benchmark::DoNotOptimize(a);
  }
}

static void bench_recursive(benchmark::State &state) {

  std::unique_ptr<Matrix> b(new Matrix());


  for (auto _ : state) {
    // transpose_recursive(*a,0,0,N);
    transpose((*a).data(),N);
    benchmark::DoNotOptimize(a);
  }
}

// Register the function as a benchmark
BENCHMARK(bench_tri)->Iterations(ITERS)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_block)->Iterations(ITERS)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_recursive)->Iterations(ITERS)->Unit(benchmark::kMillisecond);


// Run the benchmark
BENCHMARK_MAIN();