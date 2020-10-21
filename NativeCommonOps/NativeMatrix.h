#ifndef NATIVEMATRIX_H
#define NATIVEMATRIX_H

#include <Eigen/Dense>

class NativeMatrix
{
public:
    NativeMatrix();

    void resize(int numRows, int numCols);


    bool mult(NativeMatrix* a, NativeMatrix* b);

    bool multQuad(NativeMatrix* a, NativeMatrix* b);

    bool invert(NativeMatrix* a);

    bool solve(NativeMatrix* a, NativeMatrix* b);

    double* data();

    inline int rows()
    {
        return matrix.rows();
    }

    inline int cols()
    {
        return matrix.cols();
    }

private:
    Eigen::MatrixXd matrix;
};

#endif // NATIVEMATRIX_H
