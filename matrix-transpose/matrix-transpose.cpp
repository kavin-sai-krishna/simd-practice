#include "matrix-transpose.h"
#include<algorithm>
#include <immintrin.h>

void copy(Matrix &target,Matrix &source) {
    for(int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            target[i*N+j] = source[i*N+j];
        }
    }
}

void transpose_triangle(Matrix &result) {
    for(int i=0;i<N;++i) {
        for(int j=i+1;j<N;++j) {
            // std::swap(result[i][j],result[j][i]);
            std::swap(result[j*N+i],result[i*N+j]);
        }
    }
}


void transpose_block(Matrix &result) {
    for (int ii = 0; ii < N; ii += blocksize) {
        for (int jj = 0; jj < N; jj += blocksize) {
            // Determine the actual block boundaries
            int i_end = std::min(ii + blocksize, N);
            int j_end = std::min(jj + blocksize, N);

            // Iterate over the block
            for (int i = ii; i < i_end; ++i) {
                // Start j from jj to handle all elements in the block
                for (int j = jj; j < j_end; ++j) {
                    // Only swap elements in the upper triangular part of the block
                    if (i < j) {
                        std::swap(result[i*N+j], result[j*N+i]);
                    }
                }
            }
        }
    }
}

// void transpose_recursive(resultrix &result, int i, int j, int n) {
//     if (n <= 1) {
//         return;
//     }

//     // Only apply base case to diagonal subresultrices (i == j)
//     if (i == j && n <= 16) {
//         // Transpose the diagonal subresultrix
//         for (int ii = i; ii < i + n; ++ii) {
//             for (int jj = ii + 1; jj < j + n; ++jj) {
//                 std::swap(result[ii*N+jj], result[jj*N+ii]);
//             }
//         }
//         return;
//     }

//     int mid = n / 2;

//     // Recursively transpose diagonal quadrants (TL and BR)
//     transpose_recursive(result, i, j, mid);          // Top-left
//     transpose_recursive(result, i + mid, j + mid, mid); // Bottom-right

//     // Swap and transpose off-diagonal quadrants (TR and BL)
//     for (int x = 0; x < mid; ++x) {
//         for (int y = 0; y < mid; ++y) {
//             // Swap elements between TR and BL, transposing them
//             std::swap(result[(i + x)*N+(j + mid + y)], result[(i + mid + y)*N+(j + x)]);
//         }
//     }
// }

void transpose(T* result, int n) {
    if (n <= 16) {
        // Remove unused variables
        __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
        __m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

        // Load rows from result (in-place operation)
        r0 = _mm512_load_ps(&result[ 0*N]);
        r1 = _mm512_load_ps(&result[ 1*N]);
        r2 = _mm512_load_ps(&result[ 2*N]);
        r3 = _mm512_load_ps(&result[ 3*N]);
        r4 = _mm512_load_ps(&result[ 4*N]);
        r5 = _mm512_load_ps(&result[ 5*N]);
        r6 = _mm512_load_ps(&result[ 6*N]);
        r7 = _mm512_load_ps(&result[ 7*N]);
        r8 = _mm512_load_ps(&result[ 8*N]);
        r9 = _mm512_load_ps(&result[ 9*N]);
        ra = _mm512_load_ps(&result[10*N]);
        rb = _mm512_load_ps(&result[11*N]);
        rc = _mm512_load_ps(&result[12*N]);
        rd = _mm512_load_ps(&result[13*N]);
        re = _mm512_load_ps(&result[14*N]);
        rf = _mm512_load_ps(&result[15*N]);

            // ... (keep all the existing AVX-512 operations unchanged)
            // Unpack 32-bit elements
        t0 = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(r0), _mm512_castps_si512(r1)));
        t1 = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(r0), _mm512_castps_si512(r1)));
        t2 = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(r2), _mm512_castps_si512(r3)));
        t3 = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(r2), _mm512_castps_si512(r3)));
        t4 = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(r4), _mm512_castps_si512(r5)));
        t5 = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(r4), _mm512_castps_si512(r5)));
        t6 = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(r6), _mm512_castps_si512(r7)));
        t7 = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(r6), _mm512_castps_si512(r7)));
        t8 = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(r8), _mm512_castps_si512(r9)));
        t9 = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(r8), _mm512_castps_si512(r9)));
        ta = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(ra), _mm512_castps_si512(rb)));
        tb = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(ra), _mm512_castps_si512(rb)));
        tc = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(rc), _mm512_castps_si512(rd)));
        td = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(rc), _mm512_castps_si512(rd)));
        te = _mm512_castsi512_ps(_mm512_unpacklo_epi32(_mm512_castps_si512(re), _mm512_castps_si512(rf)));
        tf = _mm512_castsi512_ps(_mm512_unpackhi_epi32(_mm512_castps_si512(re), _mm512_castps_si512(rf)));

        // Unpack 64-bit elements
        r0 = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t0), _mm512_castps_si512(t2)));
        r1 = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t0), _mm512_castps_si512(t2)));
        r2 = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t1), _mm512_castps_si512(t3)));
        r3 = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t1), _mm512_castps_si512(t3)));
        r4 = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t4), _mm512_castps_si512(t6)));
        r5 = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t4), _mm512_castps_si512(t6)));
        r6 = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t5), _mm512_castps_si512(t7)));
        r7 = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t5), _mm512_castps_si512(t7)));
        r8 = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t8), _mm512_castps_si512(ta)));
        r9 = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t8), _mm512_castps_si512(ta)));
        ra = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(t9), _mm512_castps_si512(tb)));
        rb = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(t9), _mm512_castps_si512(tb)));
        rc = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(tc), _mm512_castps_si512(te)));
        rd = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(tc), _mm512_castps_si512(te)));
        re = _mm512_castsi512_ps(_mm512_unpacklo_epi64(_mm512_castps_si512(td), _mm512_castps_si512(tf)));
        rf = _mm512_castsi512_ps(_mm512_unpackhi_epi64(_mm512_castps_si512(td), _mm512_castps_si512(tf)));

        // Shuffle 128-bit lanes
        t0 = _mm512_shuffle_f32x4(r0, r4, 0x88);
        t1 = _mm512_shuffle_f32x4(r1, r5, 0x88);
        t2 = _mm512_shuffle_f32x4(r2, r6, 0x88);
        t3 = _mm512_shuffle_f32x4(r3, r7, 0x88);
        t4 = _mm512_shuffle_f32x4(r0, r4, 0xdd);
        t5 = _mm512_shuffle_f32x4(r1, r5, 0xdd);
        t6 = _mm512_shuffle_f32x4(r2, r6, 0xdd);
        t7 = _mm512_shuffle_f32x4(r3, r7, 0xdd);
        t8 = _mm512_shuffle_f32x4(r8, rc, 0x88);
        t9 = _mm512_shuffle_f32x4(r9, rd, 0x88);
        ta = _mm512_shuffle_f32x4(ra, re, 0x88);
        tb = _mm512_shuffle_f32x4(rb, rf, 0x88);
        tc = _mm512_shuffle_f32x4(r8, rc, 0xdd);
        td = _mm512_shuffle_f32x4(r9, rd, 0xdd);
        te = _mm512_shuffle_f32x4(ra, re, 0xdd);
        tf = _mm512_shuffle_f32x4(rb, rf, 0xdd);

        // Final shuffle to transpose
        r0 = _mm512_shuffle_f32x4(t0, t8, 0x88);
        r1 = _mm512_shuffle_f32x4(t1, t9, 0x88);
        r2 = _mm512_shuffle_f32x4(t2, ta, 0x88);
        r3 = _mm512_shuffle_f32x4(t3, tb, 0x88);
        r4 = _mm512_shuffle_f32x4(t4, tc, 0x88);
        r5 = _mm512_shuffle_f32x4(t5, td, 0x88);
        r6 = _mm512_shuffle_f32x4(t6, te, 0x88);
        r7 = _mm512_shuffle_f32x4(t7, tf, 0x88);
        r8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);
        r9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);
        ra = _mm512_shuffle_f32x4(t2, ta, 0xdd);
        rb = _mm512_shuffle_f32x4(t3, tb, 0xdd);
        rc = _mm512_shuffle_f32x4(t4, tc, 0xdd);
        rd = _mm512_shuffle_f32x4(t5, td, 0xdd);
        re = _mm512_shuffle_f32x4(t6, te, 0xdd);
        rf = _mm512_shuffle_f32x4(t7, tf, 0xdd);

        // Store transposed results back to original array
        _mm512_store_ps(&result[ 0*N], r0);
        _mm512_store_ps(&result[ 1*N], r1);
        _mm512_store_ps(&result[ 2*N], r2);
        _mm512_store_ps(&result[ 3*N], r3);
        _mm512_store_ps(&result[ 4*N], r4);
        _mm512_store_ps(&result[ 5*N], r5);
        _mm512_store_ps(&result[ 6*N], r6);
        _mm512_store_ps(&result[ 7*N], r7);
        _mm512_store_ps(&result[ 8*N], r8);
        _mm512_store_ps(&result[ 9*N], r9);
        _mm512_store_ps(&result[10*N], ra);
        _mm512_store_ps(&result[11*N], rb);
        _mm512_store_ps(&result[12*N], rc);
        _mm512_store_ps(&result[13*N], rd);
        _mm512_store_ps(&result[14*N], re);
        _mm512_store_ps(&result[15*N], rf);
    } else {
        int k = n / 2;
        transpose(result, k);
        transpose(result + k, k);
        transpose(result + k * N, k);       // Fixed N -> n
        transpose(result + k * N + k, k);   // Fixed N -> n
        
        // Swap off-diagonal blocks
                if (k >= 16) { // Only use AVX-512 if the block size is >= 16
            for (int i = 0; i < k; i += 16) {
                for (int j = 0; j < k; j += 16) {
                    // Load 16x16 blocks from the off-diagoNal positioNs
                    __m512 r0 = _mm512_load_ps(&result[(i + 0) * N + (j + k)]);
                    __m512 r1 = _mm512_load_ps(&result[(i + 1) * N + (j + k)]);
                    __m512 r2 = _mm512_load_ps(&result[(i + 2) * N + (j + k)]);
                    __m512 r3 = _mm512_load_ps(&result[(i + 3) * N + (j + k)]);
                    __m512 r4 = _mm512_load_ps(&result[(i + 4) * N + (j + k)]);
                    __m512 r5 = _mm512_load_ps(&result[(i + 5) * N + (j + k)]);
                    __m512 r6 = _mm512_load_ps(&result[(i + 6) * N + (j + k)]);
                    __m512 r7 = _mm512_load_ps(&result[(i + 7) * N + (j + k)]);
                    __m512 r8 = _mm512_load_ps(&result[(i + 8) * N + (j + k)]);
                    __m512 r9 = _mm512_load_ps(&result[(i + 9) * N + (j + k)]);
                    __m512 ra = _mm512_load_ps(&result[(i + 10) * N + (j + k)]);
                    __m512 rb = _mm512_load_ps(&result[(i + 11) * N + (j + k)]);
                    __m512 rc = _mm512_load_ps(&result[(i + 12) * N + (j + k)]);
                    __m512 rd = _mm512_load_ps(&result[(i + 13) * N + (j + k)]);
                    __m512 re = _mm512_load_ps(&result[(i + 14) * N + (j + k)]);
                    __m512 rf = _mm512_load_ps(&result[(i + 15) * N + (j + k)]);

                    __m512 s0 = _mm512_load_ps(&result[(i + k + 0) * N + j]);
                    __m512 s1 = _mm512_load_ps(&result[(i + k + 1) * N + j]);
                    __m512 s2 = _mm512_load_ps(&result[(i + k + 2) * N + j]);
                    __m512 s3 = _mm512_load_ps(&result[(i + k + 3) * N + j]);
                    __m512 s4 = _mm512_load_ps(&result[(i + k + 4) * N + j]);
                    __m512 s5 = _mm512_load_ps(&result[(i + k + 5) * N + j]);
                    __m512 s6 = _mm512_load_ps(&result[(i + k + 6) * N + j]);
                    __m512 s7 = _mm512_load_ps(&result[(i + k + 7) * N + j]);
                    __m512 s8 = _mm512_load_ps(&result[(i + k + 8) * N + j]);
                    __m512 s9 = _mm512_load_ps(&result[(i + k + 9) * N + j]);
                    __m512 sa = _mm512_load_ps(&result[(i + k + 10) * N + j]);
                    __m512 sb = _mm512_load_ps(&result[(i + k + 11) * N + j]);
                    __m512 sc = _mm512_load_ps(&result[(i + k + 12) * N + j]);
                    __m512 sd = _mm512_load_ps(&result[(i + k + 13) * N + j]);
                    __m512 se = _mm512_load_ps(&result[(i + k + 14) * N + j]);
                    __m512 sf = _mm512_load_ps(&result[(i + k + 15) * N + j]);

                    // Store swapped blocks
                    _mm512_store_ps(&result[(i + k + 0) * N + j], r0);
                    _mm512_store_ps(&result[(i + k + 1) * N + j], r1);
                    _mm512_store_ps(&result[(i + k + 2) * N + j], r2);
                    _mm512_store_ps(&result[(i + k + 3) * N + j], r3);
                    _mm512_store_ps(&result[(i + k + 4) * N + j], r4);
                    _mm512_store_ps(&result[(i + k + 5) * N + j], r5);
                    _mm512_store_ps(&result[(i + k + 6) * N + j], r6);
                    _mm512_store_ps(&result[(i + k + 7) * N + j], r7);
                    _mm512_store_ps(&result[(i + k + 8) * N + j], r8);
                    _mm512_store_ps(&result[(i + k + 9) * N + j], r9);
                    _mm512_store_ps(&result[(i + k + 10) * N + j], ra);
                    _mm512_store_ps(&result[(i + k + 11) * N + j], rb);
                    _mm512_store_ps(&result[(i + k + 12) * N + j], rc);
                    _mm512_store_ps(&result[(i + k + 13) * N + j], rd);
                    _mm512_store_ps(&result[(i + k + 14) * N + j], re);
                    _mm512_store_ps(&result[(i + k + 15) * N + j], rf);

                    _mm512_store_ps(&result[(i + 0) * N + (j + k)], s0);
                    _mm512_store_ps(&result[(i + 1) * N + (j + k)], s1);
                    _mm512_store_ps(&result[(i + 2) * N + (j + k)], s2);
                    _mm512_store_ps(&result[(i + 3) * N + (j + k)], s3);
                    _mm512_store_ps(&result[(i + 4) * N + (j + k)], s4);
                    _mm512_store_ps(&result[(i + 5) * N + (j + k)], s5);
                    _mm512_store_ps(&result[(i + 6) * N + (j + k)], s6);
                    _mm512_store_ps(&result[(i + 7) * N + (j + k)], s7);
                    _mm512_store_ps(&result[(i + 8) * N + (j + k)], s8);
                    _mm512_store_ps(&result[(i + 9) * N + (j + k)], s9);
                    _mm512_store_ps(&result[(i + 10) * N + (j + k)], sa);
                    _mm512_store_ps(&result[(i + 11) * N + (j + k)], sb);
                    _mm512_store_ps(&result[(i + 12) * N + (j + k)], sc);
                    _mm512_store_ps(&result[(i + 13) * N + (j + k)], sd);
                    _mm512_store_ps(&result[(i + 14) * N + (j + k)], se);
                    _mm512_store_ps(&result[(i + 15) * N + (j + k)], sf);
                }
            }
        } else {
            printf("AAAAAAa");
            // Fallback to scalar swapping for small blocks
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    std::swap(result[i * n + (j + k)], result[(i + k) * n + j]);
                }
            }
        }

        // Handle odd-sized matrices (same as before)
        if (n & 1) {
            printf("bbbbbbbbb");
            const int last = n - 1;
            for (int i = 0; i < last; i++) {
                std::swap(result[i * n + last], result[last * n + i]);
            }
        }
    }
}