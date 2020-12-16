#include "NativeSparseMatrix.h"
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>
#include <cstring>

NativeSparseMatrixImpl::NativeSparseMatrixImpl(int numRows, int numCols) : data(numRows, numCols)
{
}

void NativeSparseMatrixImpl::resize(int numRows, int numCols)
{
    if(numRows == rows() && numCols == cols())
    {
        return;
    }

    data.resize(numRows, numCols);
}

bool NativeSparseMatrixImpl::set(NativeSparseMatrixImpl *a)
{
    resize(a->rows(), a->cols());

    data = a->data;

    return true;
}

bool NativeSparseMatrixImpl::add(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->rows() || a->cols() != b->cols())
    {
        return false;
    }

    resize(a->rows(), a->cols());

    data = (a->data) + (b->data);

    return true;
}

bool NativeSparseMatrixImpl::subtract(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->rows() || a->cols() != b->cols())
    {
        return false;
    }
    resize(a->rows(), a->cols());

    data = (a->data) - (b->data);

    return true;
}


bool NativeSparseMatrixImpl::mult(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->rows())
    {
        return false;
    }

    resize(a->rows(), b->cols());

    data = (a->data) * (b->data);

    return true;
}

bool NativeSparseMatrixImpl::mult(double scale, NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->rows())
    {
        return false;
    }

    resize(a->rows(), b->cols());

    data = scale * (a->data) * (b->data);

    return true;
}

bool NativeSparseMatrixImpl::multAdd(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    data += (a->data) * (b->data);

    return true;
}

bool NativeSparseMatrixImpl::multTransA(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if( a->rows() != b->rows())
    {
        return false;
    }

    resize(a->cols(), b->cols());

    data = (a->data.transpose()) * (b->data);

    return true;
}

bool NativeSparseMatrixImpl::multAddTransA(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != rows() || b->cols() != cols() || a->rows() != b->rows())
    {
        return false;
    }

    data += (a->data.transpose()) * (b->data);

    return true;
}

bool NativeSparseMatrixImpl::multTransB(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->cols() != b->cols())
    {
        return false;
    }

    resize(a->rows(), b->rows());

    data = (a->data) * (b->data.transpose());

    return true;
}

bool NativeSparseMatrixImpl::multAddTransB(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != rows() || b->rows() != cols() || a->cols() != b->cols())
    {
        return false;
    }

    data += (a->data) * (b->data.transpose());

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

    int rowOffset = destStartRow - srcStartRow;
    int colOffset = destStartColumn - srcStartColumn;

    for (int k = 0; k < a->data.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(a->data, k); it; ++it)
        {
            data.coeffRef(rowOffset + it.row(), colOffset + it.col()) += scale * it.value();
        }
    }

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

    Eigen::SparseMatrix<double> res = a->data * b->data;

    for (int k = 0; k < res.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(res, k); it; ++it)
        {
            data.coeffRef(rowStart + it.row(), colStart + it.col()) += it.value();
        }
    }

    return true;

}

bool NativeSparseMatrixImpl::multQuad(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{
    if(a->rows() != b->cols() || b->cols() != b->rows())
    {
        return false;
    }

    resize(a->cols(), a->cols());

    data = (a->data).transpose() * (b->data) * (a->data);

    return true;
}

bool NativeSparseMatrixImpl::invert(NativeSparseMatrixImpl *a)
{
    if(a->rows() != a->cols())
    {
        return false;
    }

    resize(a->rows(), a->cols());

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    Eigen::SparseMatrix<double> A = a->data;
    A.makeCompressed();
    Eigen::SparseMatrix<double> identity(a->data.rows(), a->data.rows());
    identity.setIdentity();
    // solver.compute(a->data);
    solver.analyzePattern(A);
    solver.factorize(A);
    data = solver.solve(identity);;

    return true;
}

bool NativeSparseMatrixImpl::solve(NativeSparseMatrixImpl *a, NativeSparseMatrixImpl *b)
{

    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows())
    {
        return false;
    }

    resize(a->cols(), 1);

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(a->data);
    data = (solver.solve(b->data));

    return true;

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


    Eigen::SparseMatrix<double> val = src->data.block(srcY0, srcX0, h, w);
    for (int k = 0; k < val.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(src->data, k); it; ++it)
        {
            data.coeffRef(dstY0 + it.row(), dstX0 + it.col()) = it.value();
        }
    }

    return true;
}

bool NativeSparseMatrixImpl::insert(int *srcColIndexPtr, int *srcNzRowPtr, double *srcValuePtr, int srcRows, int srcCols, int srcNnz, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
{
    if(srcColIndexPtr == nullptr || srcNzRowPtr == nullptr || srcValuePtr == nullptr)
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

    if (srcNnz == 0)
    {
        return true;
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

    int rowOffset = dstY0-srcY0;
    int colOffset = dstX0-srcX0;

    for (int col = srcX0; col < srcX1 + 1; col++)
    {
        int startDataRow = srcColIndexPtr[col];
        int endDataRow = srcColIndexPtr[col + 1];


        for (int dataRow = startDataRow; dataRow < endDataRow; dataRow++)
        {
            int row = srcNzRowPtr[dataRow];
            if (row < srcY0 || row > srcY1)
                continue;

            std::cout << "source " << row << ", " << col << " = " << srcValuePtr[dataRow] << std::endl;
            std::cout << "data row " << dataRow << std::endl;
            std::cout << "dst " << row + rowOffset << ", " << col + colOffset << std::endl;
            data.coeffRef(row + rowOffset, col + colOffset) = srcValuePtr[dataRow];
        }
    }
    

    return true;
}

bool NativeSparseMatrixImpl::extract(int srcY0, int srcY1, int srcX0, int srcX1, int *dstColIndexPtr, int *dstNzRowPtr, double *dstValuePtr, int *dstNnz, int dstRows, int dstCols, int dstY0, int dstX0)
{
    if(dstColIndexPtr == nullptr || dstNzRowPtr == nullptr || dstValuePtr == nullptr || dstNnz == nullptr)
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
    Eigen::SparseMatrix<double> ref = data.block(srcY0, srcX0, h, w);

    for (int k = 0; k < ref.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(ref, k); it; ++it)
        {
            eigenData.coeffRef(dstY0 + it.row(), dstX0 + it.col()) = it.value();
        }
    }

    return true;

}


bool NativeSparseMatrixImpl::transpose(NativeSparseMatrixImpl *a)
{
    resize(a->cols(), a->rows());

    data = a->data.transpose();

    return true;
}

bool NativeSparseMatrixImpl::containsNaN()
{
    for(int i = 0; i < data.nonZeros(); i++)
    {
        if(std::isnan(data.valuePtr()[i]))
        {
            return true;
        }
    }

    return false;
}

bool NativeSparseMatrixImpl::scale(double scale, NativeSparseMatrixImpl *src)
{
    resize(src->rows(), src->cols());

    data = scale * src->data;

    return true;
}

bool NativeSparseMatrixImpl::isAprrox(NativeSparseMatrixImpl *other, double precision)
{
    return data.isApprox(other->data, precision);
}
 
bool NativeSparseMatrixImpl::set(double *valuePtr, int *nz_rows, int *col_idx, int rows, int cols, int nnz)
{
    if(valuePtr == nullptr || nz_rows == nullptr || col_idx == nullptr)
    {
        return false;
    }

    resize(rows, cols);

    std::vector<T> values;
    values.reserve(nnz);
    for (int col = 0; col < cols; col++)
    {
        for (int idx = col_idx[col]; idx < col_idx[col + 1]; idx++)
        {
            values.push_back(T(nz_rows[idx], col, valuePtr[idx]));
        }
    }

    data.setZero();
    data.setFromTriplets(values.begin(), values.end());


    return true;

}

bool NativeSparseMatrixImpl::get(double *data, int *nz_rows, int *col_idx, int rows, int cols, int *nnz)
{
    if(rows != this->rows() || cols != this->cols())
    {
        return false;
    }

    if(data == nullptr || nz_rows == nullptr || col_idx == nullptr || nnz == nullptr)
    {
        return false;
    }

    this->data.makeCompressed();
    nnz[0] = this->data.nonZeros();
    for (int i = 0; i < this->data.nonZeros(); i++)
    {
        data[i] = this->data.valuePtr()[i];
        nz_rows[i] = this->data.innerIndexPtr()[i];
    }
    for (int i = 0; i < cols + 1; i++)
    {
        col_idx[i] = this->data.outerIndexPtr()[i];
    }

    return true;
}

void NativeSparseMatrixImpl::print()
{
    std::cout << data << std::endl;
}






