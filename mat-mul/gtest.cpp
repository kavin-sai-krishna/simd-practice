#include "mat-mul.h"
#include <gtest/gtest.h>
#include<cmath>

namespace {
    static bool equals(const Matrix &a, const Matrix &b) {
        constexpr int maxErrors = 10;
        const float epsilon = std::sqrt(std::numeric_limits<float>::epsilon());

        int errors = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
            T va = a[i*N+j];
            T vb = b[i*N+j];
            float error = std::abs(va - vb);
            if (error >= epsilon) {
                // std::cerr << "Result[" << i << ", " << j << "] = " << va
                //         << ". Expected[" << i << ", " << j << "] = " << vb
                //         << std::endl;
                if (++errors >= maxErrors)
                return false;
            }
            }
        }
        return 0 == errors;
    }
}

TEST(mul_check,naive_optimized) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());
    std::unique_ptr<Matrix> c(new Matrix());
    std::unique_ptr<Matrix> d(new Matrix());

    init(*a);
    init(*b); // b=a
    naive_for(*c,*a,*b);
    optimized_for(*d,*a,*b);
    EXPECT_TRUE(equals(*c,*d));
}

TEST(mul_check,transpose_mul) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());
    std::unique_ptr<Matrix> c(new Matrix());
    std::unique_ptr<Matrix> d(new Matrix());

    init(*a);
    init(*b); // b=a
    naive_for(*c,*a,*b);
    transpose_mul(*d,*a,*b);
    EXPECT_TRUE(equals(*c,*d));
}

