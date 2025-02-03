#include "mat-mul.h"
#include <memory>
#include<immintrin.h>

void naive_for(Matrix &result,const Matrix& a,const Matrix& b) {
    for(int i=0;i<N;++i) {
        for(int j=0;j<N;++j) {
            for(int k=0;k<N;++k) {
                result[i*N+j] += a[i*N+k] * b[k*N+j];
            }
        }
    }
}
void optimized_for(Matrix &result,const Matrix& a,const Matrix& b) {
    for(int k=0;k<N;++k) {
        for(int i=0;i<N;++i) {
            for(int j=0;j<N;++j) {
                result[k*N+j] += a[k*N+i]*b[i*N+j];
            }
        }
    }
}

void transpose_mul(Matrix &result,const Matrix& a,Matrix& b) {
    transpose_block(b);
    for(int i=0;i<N;++i) {
        for(int j=0;j<N;++j) {
            for(int k=0;k<N;++k) {
                result[i*N+j] += a[i*N+k] * b[j*N+k];
            }
        }
    }
    transpose_block(b);
    
}

void avx_multiply(Matrix &result, const Matrix &a, const Matrix &b) {
    const int BLOCK3 = 64, BLOCK2 = 120, BLOCK1 = 240;
    std::memset(&result, 0, N*N*sizeof(float));

    for (int i3 = 0; i3 < N; i3 += BLOCK3) {
        int end_i3 = std::min(i3 + BLOCK3, N);
        for (int i2 = 0; i2 < N; i2 += BLOCK2) {
            int end_i2 = std::min(i2 + BLOCK2, N);
            for (int i1 = 0; i1 < N; i1 += BLOCK1) {
                int end_i1 = std::min(i1 + BLOCK1, N);
                for (int x = i2; x < end_i2; x += 6) {
                    for (int y = i3; y < end_i3; y += 16) {

                        __m256 c00 = _mm256_loadu_ps(&result[x*N + y]);
                        __m256 c01 = _mm256_loadu_ps(&result[x*N + y + 8]);
                        __m256 c10 = _mm256_loadu_ps(&result[(x+1)*N + y]);
                        __m256 c11 = _mm256_loadu_ps(&result[(x+1)*N + y + 8]);
                        __m256 c20 = _mm256_loadu_ps(&result[(x+2)*N + y]);
                        __m256 c21 = _mm256_loadu_ps(&result[(x+2)*N + y + 8]);
                        __m256 c30 = _mm256_loadu_ps(&result[(x+3)*N + y]);
                        __m256 c31 = _mm256_loadu_ps(&result[(x+3)*N + y + 8]);
                        __m256 c40 = _mm256_loadu_ps(&result[(x+4)*N + y]);
                        __m256 c41 = _mm256_loadu_ps(&result[(x+4)*N + y + 8]);
                        __m256 c50 = _mm256_loadu_ps(&result[(x+5)*N + y]);
                        __m256 c51 = _mm256_loadu_ps(&result[(x+5)*N + y + 8]);

                        for (int k = i1; k < end_i1; ++k) {
                            __m256 b0 = _mm256_loadu_ps(&b[k*N + y]);
                            __m256 b1 = _mm256_loadu_ps(&b[k*N + y + 8]);

                            __m256 a0 = _mm256_broadcast_ss(&a[x*N + k]);
                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);

                            __m256 a1 = _mm256_broadcast_ss(&a[(x+1)*N + k]);
                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);

                            __m256 a2 = _mm256_broadcast_ss(&a[(x+2)*N + k]);
                            c20 = _mm256_fmadd_ps(a2, b0, c20);
                            c21 = _mm256_fmadd_ps(a2, b1, c21);

                            __m256 a3 = _mm256_broadcast_ss(&a[(x+3)*N + k]);
                            c30 = _mm256_fmadd_ps(a3, b0, c30);
                            c31 = _mm256_fmadd_ps(a3, b1, c31);

                            __m256 a4 = _mm256_broadcast_ss(&a[(x+4)*N + k]);
                            c40 = _mm256_fmadd_ps(a4, b0, c40);
                            c41 = _mm256_fmadd_ps(a4, b1, c41);

                            __m256 a5 = _mm256_broadcast_ss(&a[(x+5)*N + k]);
                            c50 = _mm256_fmadd_ps(a5, b0, c50);
                            c51 = _mm256_fmadd_ps(a5, b1, c51);
                        }

                        _mm256_storeu_ps(&result[x*N + y], c00);
                        _mm256_storeu_ps(&result[x*N + y + 8], c01);
                        _mm256_storeu_ps(&result[(x+1)*N + y], c10);
                        _mm256_storeu_ps(&result[(x+1)*N + y + 8], c11);
                        _mm256_storeu_ps(&result[(x+2)*N + y], c20);
                        _mm256_storeu_ps(&result[(x+2)*N + y + 8], c21);
                        _mm256_storeu_ps(&result[(x+3)*N + y], c30);
                        _mm256_storeu_ps(&result[(x+3)*N + y + 8], c31);
                        _mm256_storeu_ps(&result[(x+4)*N + y], c40);
                        _mm256_storeu_ps(&result[(x+4)*N + y + 8], c41);
                        _mm256_storeu_ps(&result[(x+5)*N + y], c50);
                        _mm256_storeu_ps(&result[(x+5)*N + y + 8], c51);
                    }
                }
            }
        }
    }
}
