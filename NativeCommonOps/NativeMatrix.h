#ifndef NATIVEMATRIX_H
#define NATIVEMATRIX_H

#include <Eigen/Dense>

class NativeMatrixImpl
{
public:
    double nan = std::numeric_limits<double>::quiet_NaN();

    NativeMatrixImpl(int numRows, int numCols);

    void resize(int numRows, int numCols);

    bool set(NativeMatrixImpl* a);

    bool add(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool subtract(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool mult(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool mult(double scale, NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multAdd(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multTransA(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multAddTransA(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multTransB(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool multAddTransB(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool addBlock(NativeMatrixImpl* a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn,
                  int numberOfRows, int numberOfColumns, double scale);


    bool multAddBlock(NativeMatrixImpl* a, NativeMatrixImpl* b, int rowStart, int colStart);

    bool multQuad(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool invert(NativeMatrixImpl* a);

    bool solve(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool solveCheck(NativeMatrixImpl* a, NativeMatrixImpl* b);

    bool insert(NativeMatrixImpl* src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0);

    bool transpose(NativeMatrixImpl* a);

    void zero();

    bool containsNaN();

    bool scale(double scale, NativeMatrixImpl* src);

    bool isAprrox(NativeMatrixImpl* other, double precision);

    double* data();

    inline double min()
    {
       return matrix.minCoeff();
    }

    inline double max()
    {
        return matrix.maxCoeff();
    }

    inline double sum()
    {
        return matrix.sum();
    }

    inline double prod()
    {
        return matrix.prod();
    }

    inline void scale(double scale)
    {
        matrix *= scale;
    }

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
