#ifndef NATIVEMATRIX_H
#define NATIVEMATRIX_H

#include <Eigen/Dense>

class NativeMatrixImpl
{
public:
    NativeMatrixImpl(int numRows, int numCols);

    void resize(int numRows, int numCols);


    bool mult(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool mult(double scale, NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multAdd(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multTransB(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multAddBlock(NativeMatrixImpl* a, NativeMatrixImpl* b, int rowStart, int colStart);

    bool multQuad(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool invert(NativeMatrixImpl* a);

    bool solve(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool solveCheck(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool insert(NativeMatrixImpl* src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0);

    void zero();

    bool scale(double scale, NativeMatrixImpl* src);

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
