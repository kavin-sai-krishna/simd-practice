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

void avx_multiply(Matrix& result, const Matrix& a, Matrix& b) {

}
