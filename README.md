# SIMD Practice Project

This project provides an implementation of matrix multiplication and matrix transpose with testing and benchmarking capabilities using GoogleTest and Google Benchmark.


## Build Instructions

Follow these steps to configure and build the project:

### 1. Clone the Repository
```sh
 git clone <repository_url>
 cd matrix-mul
 (or)
 cd matrix-transpose
```

### 2. Create a Build Directory
```sh
mkdir build
cd build
```

### 3. Configure the Project with CMake
```sh
cmake ..
```
If you prefer a specific generator, specify it:
```sh
cmake -G "Ninja" ..  # Using Ninja
cmake -G "Visual Studio 17 2022" ..  # For Visual Studio
```

### 4. Build the Project
```sh
cmake --build . --config Release
```

### 5. Run Tests
Execute the test binary directly:
```sh
./Release/VALIDATE  # Linux/macOS
Release/VALIDATE.exe  # Windows
```

### 6. Run Benchmarks
To run the benchmark executable:
```sh
./Release/BENCHMARK  # Linux/macOS
Release/BENCHMARK.exe  # Windows
```

## Notes
- If you modify CMakeLists.txt, re-run the configuration step (`cmake ..`).
- If you encounter any issues, ensure dependencies are correctly fetched and installed.
- If you want to change the array size or the number of iterations for benchmark edit N or Iterations in the (.h) file.


