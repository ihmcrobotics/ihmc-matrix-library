#include "NativeMatrix.h"
#include <Eigen/Dense>
#include <iostream>

NativeMatrixImpl::NativeMatrixImpl(int numRows, int numCols) : matrix(numRows, numCols)
{
}

void NativeMatrixImpl::resize(int numRows, int numCols)
{
    matrix.resize(numRows, numCols);
}

bool NativeMatrixImpl::mult(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix = (a->matrix) * (b->matrix);

    return true;
}

bool NativeMatrixImpl::multQuad(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != b->cols() || b->cols() != b->rows() || rows() != a->cols() || cols() != a->cols())
    {
        return false;
    }


    matrix = (a->matrix).transpose() * (b->matrix) * (a->matrix);

    return true;
}

bool NativeMatrixImpl::invert(NativeMatrixImpl *a)
{
    if(a->rows() != a->cols() || rows() != a->rows() || cols() != a->cols() )
    {
        return false;
    }

    matrix = (a->matrix).lu().inverse();

    return true;
}

bool NativeMatrixImpl::solve(NativeMatrixImpl *a, NativeMatrixImpl *b)
{

    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows() || rows() != a->cols() || cols() != 1)
    {
        return false;
    }

    matrix = (a->matrix).lu().solve((b->matrix));

    return true;

}

double *NativeMatrixImpl::data()
{
    return matrix.data();
}

void NativeMatrixImpl::print()
{
    std::cout << matrix << std::endl;
}






