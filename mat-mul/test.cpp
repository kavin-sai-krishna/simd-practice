#include <array>
#include <immintrin.h>
#include <iostream>

constexpr int N = 256;
using T = float;
using Matrix = std::array<T, N * N>;

void avx_multiply(Matrix& result, const Matrix& a, const Matrix& b) {
    for (int i = 0; i < N * N; i += 16) {  // Iterate over all elements in blocks of 16
        __m512 _a = _mm512_load_ps(&a[i]);  // Corrected pointer arithmetic
        __m512 _b = _mm512_load_ps(&b[i]);
        __m512 _c = _mm512_mul_ps(_a, _b);
        _mm512_store_ps(&result[i], _c);
    }
}

int main() {
    Matrix a, b, result;

    // Initialize matrices with some values
    for (int i = 0; i < N * N; i++) {
        a[i] = i * 0.1f;  // Example values
        b[i] = (i % 7) * 0.2f;
    }

    // Perform AVX multiplication
    avx_multiply(result, a, b);

    // Print some results for verification
    std::cout << "First 5 results:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


