#include "mat-mul.h"

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
    // for(int i=0;i<N;++i) {
    //     for(int j=i+1;j<N;++j) {
    //         std::swap(result[i*N+j],result[j*N+i]);
    //     }
    // }
}