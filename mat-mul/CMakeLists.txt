cmake_minimum_required(VERSION 3.14)

cmake_minimum_required(VERSION 3.14)

project(matrix-mul)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED YES)
set(gtest_force_shared_crt on)


include(FetchContent)

set(BENCHMARK_ENABLE_TESTING off)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

FetchContent_MakeAvailable(
    googletest
    googlebenchmark
)

add_library(
  source_code
  mat-mul.cpp
  init.cpp
  transpose.cpp
  )

  target_include_directories(
  source_code
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
enable_testing()

add_executable(VALIDATE)
target_sources(
  VALIDATE
  PRIVATE
    gtest.cpp
)
target_link_libraries(
  VALIDATE
  PRIVATE
    gtest_main
    source_code
)
include(GoogleTest)
gtest_discover_tests(VALIDATE)

add_executable(BENCHMARK)
target_sources(
  BENCHMARK
  PRIVATE
    gbench.cpp
)
target_link_libraries(
  BENCHMARK
  PRIVATE
    source_code
    benchmark::benchmark
    benchmark::benchmark_main
)

