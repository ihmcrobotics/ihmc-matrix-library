#include "NativeMatrix.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

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

bool NativeMatrixImpl::mult(double scale, NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix = scale * (a->matrix) * (b->matrix);

    return true;
}

bool NativeMatrixImpl::multAdd(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix += (a->matrix) * (b->matrix);

    return true;
}

bool NativeMatrixImpl::multTransB(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->rows() != cols() || a->cols() != b->cols())
    {
        return false;
    }

    matrix = (a->matrix) * (b->matrix.transpose());

    return true;
}

bool NativeMatrixImpl::multAddBlock(NativeMatrixImpl *a, NativeMatrixImpl *b, int rowStart, int colStart)
{
    if(a->cols() != b->rows())
    {
        return false;
    }

    if( (rows() - rowStart) < a->rows())
    {
        return false;
    }

    if((cols() - colStart) < b->cols())
    {
        return false;
    }

    matrix.block(rowStart, colStart, a->rows(), b->cols()) += a->matrix * b->matrix;

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

bool NativeMatrixImpl::solveCheck(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows() || rows() != a->cols() || cols() != 1)
    {
        std::cerr << "NativeMatrix::solveCheck: Invalid dimensions" << std::endl;
        return false;
    }

    const Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > fullPivLu = a->matrix.fullPivLu();
    if (fullPivLu.isInvertible())
    {
        matrix = fullPivLu.solve(b->matrix);
        return true;
    }
    else
    {
        matrix.setConstant(std::nan(""));
        return false;
    }

}

bool NativeMatrixImpl::insert(NativeMatrixImpl *src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
{

    if( srcY1 < srcY0 || srcY0 < 0 || srcY1 > src->rows() )
    {
        return false;
    }
    if( srcX1 < srcX0 || srcX0 < 0 || srcX1 > src->cols() )
    {
        return false;
    }

    int w = srcX1-srcX0;
    int h = srcY1-srcY0;

    if( dstY0+h > rows() )
    {
        return false;
    }
    if( dstX0+w > cols() )
    {
        return false;
    }


    matrix.block(dstY0, dstX0, h, w) = src->matrix.block(srcY0, srcX0, h, w);

    return true;
}

void NativeMatrixImpl::zero()
{
    matrix.setZero();
}

bool NativeMatrixImpl::scale(double scale, NativeMatrixImpl *src)
{
    if(cols() != src->cols() || rows() != src->rows())
    {
        return false;
    }

    matrix = scale * src->matrix;

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






