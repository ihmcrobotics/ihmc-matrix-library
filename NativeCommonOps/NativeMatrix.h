#ifndef NATIVEMATRIX_H
#define NATIVEMATRIX_H

#include <Eigen/Dense>

class NativeMatrixImpl
{
public:
    NativeMatrixImpl(int numRows, int numCols);

    void resize(int numRows, int numCols);


    bool mult(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multQuad(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool invert(NativeMatrixImpl* a);

    bool solve(NativeMatrixImpl* a, NativeMatrixImpl* b);

    double* data();

    inline int rows()
    {
        return matrix.rows();
    }

    inline int cols()
    {
        return matrix.cols();
    }

    inline int size()
    {
        return matrix.cols() * matrix.rows();
    }

    void print();

private:
    Eigen::MatrixXd matrix;
};

#endif // NATIVEMATRIX_H
