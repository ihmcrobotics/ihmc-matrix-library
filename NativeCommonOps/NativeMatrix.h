#ifndef NATIVEMATRIX_H
#define NATIVEMATRIX_H

#include <Eigen/Dense>

/**
  * Uncomment ROW_MAJOR to enable row-major mode. This might give a small performance benefit in set/get with EJML matrices.
  *
  */
//#define ROW_MAJOR

#ifdef ROW_MAJOR
#define STORAGE_FORMAT Eigen::RowMajor
#else
#define STORAGE_FORMAT Eigen::ColMajor
#endif

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, STORAGE_FORMAT>, Eigen::Aligned16> NativeMatrixView;

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

    bool removeRow(int indexToRemove);

    bool removeColumn(int indexToRemove);

    void zero();

    bool containsNaN();

    bool scale(double scale, NativeMatrixImpl* src);

    bool isAprrox(NativeMatrixImpl* other, double precision);

    bool set(double* data, int rows, int cols);

    bool get(double* data, int rows, int cols);

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


    inline bool set(int row, int col, double value)
    {
          if(row >= rows() || col >= cols())
          {
              return false;
          }

          matrix(row, col) = value;

          return true;
    }

    inline double get(int row, int col)
    {
        if(row >= rows() || col >= cols())
        {
            return nan;
        }

        return matrix(row, col);
    }

    inline int* dims()
    {
        return dimensions;
    }


    void print();

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, STORAGE_FORMAT>  storage;
    NativeMatrixView matrix;
    int dimensions[3];

    inline void updateView(int numRows, int numCols)
    {
        new (&matrix) NativeMatrixView(storage.data(), numRows, numCols);
        dimensions[0] = numRows;
        dimensions[1] = numCols;
        dimensions[2] = numRows * numCols;
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
        return matrix.size();
    }
};

#endif // NATIVEMATRIX_H
