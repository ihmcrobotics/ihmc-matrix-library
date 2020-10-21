#include "NativeMatrix.h"
#include <Eigen/Dense>


NativeMatrix::NativeMatrix()
{
}

void NativeMatrix::resize(int numRows, int numCols)
{
    matrix.resize(numRows, numCols);
}

bool NativeMatrix::mult(NativeMatrix *a, NativeMatrix *b)
{
    if(a->rows() != rows() || b->cols() != cols() || a->cols() != b->rows())
    {
        return false;
    }

    matrix = (a->matrix) * (b->matrix);

    return true;
}

bool NativeMatrix::multQuad(NativeMatrix *a, NativeMatrix *b)
{
    if(a->rows() != b->cols() || b->cols() != b->rows() || rows() != a->cols() || cols() != a->cols())
    {
        return false;
    }


    matrix = (a->matrix).transpose() * (b->matrix) * (a->matrix);

    return true;
}

bool NativeMatrix::invert(NativeMatrix *a)
{
    if(a->rows() != a->cols() || rows() != a->rows() || cols() != a->cols() )
    {
        return false;
    }

    matrix = (a->matrix).lu().inverse();

    return true;
}

bool NativeMatrix::solve(NativeMatrix *a, NativeMatrix *b)
{

    if(a->rows() != b->rows() || b->cols() != 1 || a->cols() != a->rows() || rows() != a->cols() || cols() != 1)
    {
        return false;
    }

    matrix = (a->matrix).lu().solve((b->matrix));

    return true;

}

double *NativeMatrix::data()
{
    return matrix.data();
}






