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

bool NativeMatrixImpl::set(NativeMatrixImpl *a)
{
    if(cols() != a->cols() || rows() != a->rows())
    {
        return false;
    }

    matrix = a->matrix;

    return true;
}

bool NativeMatrixImpl::add(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != cols() || b->rows() != rows())
    {
        return false;
    }

    matrix = (a->matrix) + (b->matrix);

    return true;
}

bool NativeMatrixImpl::subtract(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != cols() || b->rows() != rows())
    {
        return false;
    }

    matrix = (a->matrix) - (b->matrix);

    return true;
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

bool NativeMatrixImpl::multTransA(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->cols() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix = (a->matrix.transpose()) * (b->matrix);

    return true;
}

bool NativeMatrixImpl::multAddTransA(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->cols() != rows() || b->cols() != cols() || a->rows() != b->rows())
    {
        return false;
    }

    matrix += (a->matrix.transpose()) * (b->matrix);

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

bool NativeMatrixImpl::multAddTransB(NativeMatrixImpl *a, NativeMatrixImpl *b)
{
    if(a->rows() != rows() || b->rows() != cols() || a->cols() != b->cols())
    {
        return false;
    }

    matrix += (a->matrix) * (b->matrix.transpose());

    return true;
}

bool NativeMatrixImpl::addBlock(NativeMatrixImpl *a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns, double scale)
{
    if(rows() < destStartRow + numberOfRows)
    {
        return false;
    }

    if(cols() < destStartColumn + numberOfColumns)
    {
        return false;
    }

    if(a->rows() < srcStartRow + numberOfRows)
    {
        return false;
    }

    if(a->cols() < srcStartColumn + numberOfColumns)
    {
        return false;
    }

    matrix.block(destStartRow, destStartColumn, numberOfRows, numberOfColumns) += scale * a->matrix.block(srcStartRow, srcStartColumn, numberOfRows, numberOfColumns);
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
        matrix.setConstant(nan);
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

bool NativeMatrixImpl::containsNaN()
{
    for(int i = 0; i < matrix.size(); i++)
    {
        if(std::isnan(matrix.data()[i]))
        {
            return true;
        }
    }

    return false;
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

bool NativeMatrixImpl::isAprrox(NativeMatrixImpl *other, double precision)
{
    return matrix.isApprox(other->matrix, precision);
}

double *NativeMatrixImpl::data()
{
    return matrix.data();
}

void NativeMatrixImpl::print()
{
    std::cout << matrix << std::endl;
}






