#pragma once
#include<array>

constexpr int N = 1056;
constexpr int blocksize = 16;
constexpr int ITERS = 25;
using T = float;
using Matrix = std::array<T,N*N>;

void naive_for(Matrix &result,const Matrix& a,const Matrix& b);
void optimized_for(Matrix &result,const Matrix& a,const Matrix& b);
void transpose_mul(Matrix &result,const Matrix& a,Matrix& b);
void init(Matrix &matrix);
void copy(Matrix &target,Matrix &source);
// void transpose_recursive(Matrix &result,int i,int j,int n);
void transpose_block(Matrix &result);
void avx_multiply(Matrix &result, const Matrix &a, const Matrix &b);