#include "matrix-transpose.h"
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
                std::cerr << "Result[" << i << ", " << j << "] = " << va
                        << ". Expected[" << i << ", " << j << "] = " << vb
                        << std::endl;
                if (++errors >= maxErrors)
                return false;
            }
            }
        }
        return 0 == errors;
    }
}

TEST(Transpose_check,uppertriangle) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());

    init(*a);
    copy(*b,*a); // b=a
    transpose_triangle(*a);
    transpose_triangle(*a);
    EXPECT_TRUE(equals(*a,*b));
}

TEST(Transpose_check,blockmatrix) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());

    init(*a);
    copy(*b,*a);// b = a 
    transpose_block(*a);
    transpose_block(*a);

    EXPECT_TRUE(equals(*a,*b));
}

TEST(Transpose_check_false_case,uppertriangle) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());

    init(*a);
    copy(*b,*a); // b=a
    transpose_triangle(*a);
    EXPECT_FALSE(equals(*a,*b));
}

TEST(Transpose_check_false_case,block) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());

    init(*a);
    copy(*b,*a); // b=a
    transpose_block(*a);
    EXPECT_FALSE(equals(*a,*b));
}

TEST(MAIN,block) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());

    init(*a);
    copy(*b,*a); // b=a
    transpose_block(*a);
    transpose_triangle(*b);

    EXPECT_TRUE(equals(*a,*b));

}

TEST(MAIN,recursive) {
    std::unique_ptr<Matrix> a(new Matrix());
    std::unique_ptr<Matrix> b(new Matrix());


    init(*a);
    copy(*b,*a); // b=a
    transpose_triangle(*a);
    // transpose_recursive(*b,0,0,N);
    transpose((*b).data(),N);
    
    EXPECT_EQ(equals(*a,*b),true);
}
