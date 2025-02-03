#include "matrix-transpose.h"
#include <cmath>
#include <limits>
#include <random>

// void init(Matrix &matrix) {
//   std::default_random_engine generator;
//   std::uniform_real_distribution<T> distribution(static_cast<T>(-1000),static_cast<T>(1000));

//   for (int i = 0; i < N; i++) {
//     T sum = 0;
//     for (int j = 0; j < N; j++) {
//       T value = distribution(generator);
//       sum += value * value;
//       matrix[i*N+j] = value;
//     }

//     // Normalize rows
//     if (sum >= std::numeric_limits<T>::min()) {
//       T scale = 1.0f / std::sqrt(sum);
//       for (int j = 0; j < N; j++) {
//         matrix[i*N+j] *= scale;
//       }
//     }
//   }
// }

void init(Matrix& matrix) {
  T k=1;
  for(int i=0;i<N;++i) {
    for(int j=0;j<N;++j) {
      matrix[i*N+j] = k;
      ++k;
    }
  }
}