#include "NativeSparseMatrix.h"
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>
#include <cstring>

NativeSparseMatrixImpl::NativeSparseMatrixImpl(int numRows, int numCols) : storage(numRows, numCols), matrix(NULL, numRows, numCols)
{
    updateView(numRows, numCols);
}

void NativeSparseMatrixImpl::resize(int numRows, int numCols)
{
    if(numRows == rows() && numCols == cols())
    {
        return;
    }

    if(numRows * numCols > storage.size())
    {
        storage.resize(numRows, numCols);
    }

    updateView(numRows, numCols);
}

bool NativeSparseMatrixImpl::set(NativeSparseMatrixImpl *a)
{
    resize(a->rows(), a->cols());

    matrix = a->matrix;

    return true;
}

bool NativeSparseMatrixImpl::add(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->rows() || a->cols() != b->cols())
    {
        return false;
    }

    resize(a->rows(), a->cols());

    matrix = (a->matrix) + (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::subtract(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->rows() || a->cols() != b->cols())
    {
        return false;
    }
    resize(a->rows(), a->cols());

    matrix = (a->matrix) - (b->matrix);

    return true;
}


bool NativeSparseMatrixImpl::mult(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->rows())
    {
        return false;
    }

    resize(a->rows(), b->cols());

    matrix = (a->matrix) * (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::mult(double scale, NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->rows())
    {
        return false;
    }

    resize(a->rows(), b->cols());

    matrix = scale * (a->matrix) * (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::multAdd(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix += (a->matrix) * (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::multTransA(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if( a->rows() != b->rows())
    {
        return false;
    }

    resize(a->cols(), b->cols());

    matrix = (a->matrix.transpose()) * (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::multAddTransA(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != rows() || b->cols() != cols() || a->rows() != b->rows())
    {
        return false;
    }

    matrix += (a->matrix.transpose()) * (b->matrix);

    return true;
}

bool NativeSparseMatrixImpl::multTransB(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->cols())
    {
        return false;
    }

    resize(a->rows(), b->rows());

    matrix = (a->matrix) * (b->matrix.transpose());

    return true;
}

bool NativeSparseMatrixImpl::multAddTransB(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != rows() || b->rows() != cols() || a->cols() != b->cols())
    {
        return false;
    }

    matrix += (a->matrix) * (b->matrix.transpose());

    return true;
}

bool NativeSparseMatrixImpl::addBlock(NativeSparseMatrixImpl *a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns, double scale)
{
    if(destStartRow < 0 || destStartColumn < 0 || srcStartRow < 0 || srcStartColumn < 0 || numberOfRows < 0 || numberOfColumns < 0)
    {
        return false;
    }

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

bool NativeSparseMatrixImpl::multAddBlock(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b, int rowStart, int colStart)
{
    if(rowStart < 0 || colStart < 0)
    {
        return false;
    }

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

bool NativeSparseMatrixImpl::multQuad(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->cols() || b->cols() != b->rows())
    {
        return false;
    }

    resize(a->cols(), a->cols());

    matrix = (a->matrix).transpose() * (b->matrix) * (a->matrix);

    return true;
}

bool NativeSparseMatrixImpl::invert(NativeSparseMatrixImpl *a)
{
    if(a->rows() != a->cols())
    {
        return false;
    }

    resize(a->rows(), a->cols());

    matrix = (a->matrix).lu().inverse();

    return true;
}

bool NativeSparseMatrixImpl::solve(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{

    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows())
    {
        return false;
    }

    resize(a->cols(), 1);

    matrix = (a->matrix).lu().solve((b->matrix));

    return true;

}

bool NativeSparseMatrixImpl::solveCheck(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows())
    {
        std::cerr << "NativeSparseMatrix::solveCheck: Invalid dimensions" << std::endl;
        return false;
    }

    resize(a->cols(), 1);

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

bool NativeSparseMatrixImpl::insert(NativeSparseMatrixImpl *src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
{
    if(srcY0 < 0 || srcY1 < 0 || srcX0 < 0 || srcX1 < 0 || dstY0 < 0 || dstX0 < 0)
    {
        return false;
    }


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

bool NativeSparseMatrixImpl::insert(int *srcColIndexPtr, int *srcNzRowPtr, double *srcValuePtr, int srcRows, int srcCols, int srcNnz, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
{
    if(src == nullptr)
    {
        return false;
    }

    if(srcY0 < 0 || srcY1 < 0 || srcX0 < 0 || srcX1 < 0 || dstY0 < 0 || dstX0 < 0)
    {
        return false;
    }

    if( srcY1 < srcY0 || srcY0 < 0 || srcY1 > srcRows )
    {
        return false;
    }
    if( srcX1 < srcX0 || srcX0 < 0 || srcX1 > srcCols )
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

    Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> eigenData(srcRows, srcCols, srcNnz, srcColIndexPtr, srcNzRowPtr, srcValuePtr);
    matrix.block(dstY0, dstX0, h, w) = eigenData.block(srcY0, srcX0, h, w);

    return true;

}

bool NativeSparseMatrixImpl::extract(int srcY0, int srcY1, int srcX0, int srcX1, int *dstColIndexPtr, int *dstNzRowPtr, double *dstValuePtr, int *dstNnz, int dstRows, int dstCols, int dstY0, int dstX0)
{
    if(dst == nullptr)
    {
        return false;
    }

    if(srcY0 < 0 || srcY1 < 0 || srcX0 < 0 || srcX1 < 0 || dstY0 < 0 || dstX0 < 0)
    {
        return false;
    }

    if( srcY1 < srcY0 || srcY0 < 0 || srcY1 > rows() )
    {
        return false;
    }
    if( srcX1 < srcX0 || srcX0 < 0 || srcX1 > cols() )
    {
        return false;
    }

    int w = srcX1-srcX0;
    int h = srcY1-srcY0;

    if( dstY0+h > dstRows )
    {
        return false;
    }
    if( dstX0+w > dstCols )
    {
        return false;
    }

    Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> eigenData(dstRows, dstCols, dstNnz[0], dstColIndexPtr, dstNzRowPtr, dstValuePtr);

    eigenData.block(dstY0, dstX0, h, w) = matrix.block(srcY0, srcX0, h, w);

    return true;

}


bool NativeSparseMatrixImpl::transpose(NativeSparseMatrixImpl *a)
{
    resize(a->cols(), a->rows());

    matrix = a->matrix.transpose();

    return true;
}

/*
bool NativeSparseMatrixImpl::removeRow(int rowToRemove)
{

    if(rowToRemove >= rows() || rowToRemove < 0)
    {
        return false;
    }

    if(rows() <= 1)
    {
        updateView(0, cols());
        return true;
    }

    int oldRows = rows();
    int newRows = oldRows - 1;
    int newCols = cols();
    */

    /*
     * Algorithm based on memmove
     *
     * Very fast compared to eigen directly.
     */

/*
    double* data = storage.data();

    size_t newStride = (size_t)newRows * sizeof(double);


    for (int col = 0; col < newCols - 1; col++)
    {
        double* dst = data + (col * newRows + rowToRemove);
        double* src = data + (col * oldRows + rowToRemove + 1);

        memmove((void*)(dst), (void*)(src), newStride);
    }

    int lastCol = cols() - 1;
    int remaining = newRows - rowToRemove;
    double* dst = data + (lastCol * newRows + rowToRemove);
    double* src = data + (lastCol * oldRows + rowToRemove + 1);
    memmove((void*)(dst), (void*)(src), remaining * sizeof(double));

    updateView(newRows, newCols);
    return true;


}

bool NativeSparseMatrixImpl::removeColumn(int colToRemove)
{
    if(colToRemove >= cols() || colToRemove < 0)
    {
        return false;
    }

    if(cols() <= 1)
    {
        updateView(rows(), 0);
        return true;
    }


    int newRows = rows();
    int oldCols = cols();
    int newCols = oldCols - 1;

    double* data = storage.data();
    double* dst = data + (colToRemove * newRows);
    double* src = data + ( (colToRemove + 1) * newRows);
    size_t size = (newCols - colToRemove) * newRows * sizeof(double);

    memmove(dst, src, size);

    updateView(newRows, newCols);
    return true;
}
*/

void NativeSparseMatrixImpl::zero()
{
    matrix.setZero();
}

bool NativeSparseMatrixImpl::containsNaN()
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

bool NativeSparseMatrixImpl::scale(double scale, NativeSparseMatrixImpl *src)
{
    resize(src->rows(), src->cols());

    matrix = scale * src->matrix;

    return true;
}

bool NativeSparseMatrixImpl::isAprrox(NativeSparseMatrixImpl *other, double precision)
{
    return matrix.isApprox(other->matrix, precision);
}

bool NativeSparseMatrixImpl::set(double *valuePtr, int *nz_rows, int *col_idx, int rows, int cols, int nnz)
{
    if(data == nullptr)
    {
        return false;
    }

    resize(rows, cols);


    Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> eigenData(rows, cols, nnz, col_idx, nz_rows, valuePtr);
    matrix = eigenData;

    return true;

}

bool NativeSparseMatrixImpl::get(double *data, int *nz_rows, int *col_idx, int rows, int cols, int *nnz)
{
    if(rows != this->rows() || cols != this->cols())
    {
        return false;
    }

    if(data == nullptr)
    {
        return false;
    }

    Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> eigenData(rows, cols, nnz[0], col_idx, nz_rows, data);
    eigenData = matrix;

    return true;
}

void NativeSparseMatrixImpl::print()
{
    std::cout << matrix << std::endl;
}






