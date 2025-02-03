#pragma once
#include<array>

constexpr int N = 1024;
static constexpr int blocksize = 8;
constexpr int ITERS = 1000;
using T = float;
using Matrix = std::array<T,N*N>;

void transpose_triangle(Matrix &result);
void transpose_block(Matrix &result);
void init(Matrix &matrix);
void copy(Matrix &target,Matrix &source);
// void transpose_recursive(Matrix &result,int i,int j,int n);
void transpose(T* result, int n);