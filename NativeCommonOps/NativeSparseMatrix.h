#ifndef NATIVESPARSEMATRIX_H
#define NATIVESPARSEMATRIX_H

#include <Eigen/Sparse>


typedef Eigen::Map<Eigen::SparseMatrix<double>, Eigen::AlignedMax> NativeSparseMatrixView;
typedef Eigen::Triplet<double> T;

class NativeSparseMatrixImpl
{
public:
    double nan = std::numeric_limits<double>::quiet_NaN();

    NativeSparseMatrixImpl(int numRows, int numCols);

    void resize(int numRows, int numCols);

    bool set(NativeSparseMatrixImpl* a);

    bool add(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool subtract(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool mult(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool mult(double scale, NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool multAdd(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool multTransA(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool multAddTransA(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool multTransB(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool multAddTransB(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool addBlock(NativeSparseMatrixImpl* a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn,
                  int numberOfRows, int numberOfColumns, double scale);


    bool multAddBlock(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b, int rowStart, int colStart);

    bool multQuad(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool invert(NativeSparseMatrixImpl* a);

    bool solve(NativeSparseMatrixImpl* a, NativeSparseMatrixImpl* b);

    bool insert(NativeSparseMatrixImpl* src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0);

    bool insert(int *srcColIdexPtr, int *srcNzRowPtr, double *srcValuePtr, int srcRows, int srcCols, int srcNnz, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0);

    bool extract(int srcY0, int srcY1, int srcX0, int srcX1, int *dstColIndexPtr, int *dstNzRowPtr, double *dstValuePtr, int *dstNnz, int dstRows, int dstCols, int dstY0, int dstX0);

    bool transpose(NativeSparseMatrixImpl* a);


    void zero();

    bool containsNaN();

    bool scale(double scale, NativeSparseMatrixImpl* src);

    bool isAprrox(NativeSparseMatrixImpl* other, double precision);

    bool set(double* data, int* nz_rows, int* col_idx, int rows, int cols, int nnz);

    bool get(double* data, int* nz_rows, int* col_idx, int rows, int cols, int* nnz);

    inline double sum()
    {
        return data.sum();
    }

    inline void scale(double scale)
    {
        data *= scale;
    }


    inline bool set(int row, int col, double value)
    {
          if(row >= rows() || col >= cols() || row < 0 || col < 0)
          {
              return false;
          }

          // FIXME is this right?
          data.coeffRef(row, col) = value;

          return true;
    }

    inline double get(int row, int col)
    {
        if(row >= rows() || col >= cols() || row < 0 || col < 0)
        {
            return nan;
        }

        // FIXME is this right?
        return data.coeffRef(row, col);
    }


    inline int rows()
    {
        return data.rows();
    }

    inline int cols()
    {
        return data.cols();
    }

    inline int size()
    {
        return data.size();
    }

    inline int nonZeros()
    {
        return data.nonZeros();
    }


    void print();

    NativeSparseMatrixView matrix;


private:
    Eigen::SparseMatrix<double> data;


    inline void updateView(int numRows, int numCols)
    {
        new (&matrix) NativeSparseMatrixView(numRows, numCols, data.nonZeros(), data.innerIndexPtr(), data.outerIndexPtr(), data.valuePtr());
    }

};

#endif // NATIVESPARSEMATRIX_H
