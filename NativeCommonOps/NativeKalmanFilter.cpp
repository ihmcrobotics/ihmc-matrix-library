#include "NativeKalmanFilter.h"
#include <iostream>
using Eigen::MatrixXd;


NativeKalmanFilterImpl::NativeKalmanFilterImpl()
{

}

bool NativeKalmanFilterImpl::predictErrorCovariance(NativeMatrixImpl *errorCovariance, NativeMatrixImpl *F, NativeMatrixImpl *P, NativeMatrixImpl *Q)
{
    {

        if (F->cols() != P->rows() || P->rows() != P->cols() || F->rows() != Q->rows() || Q->rows() != Q->cols() || F->cols() != F->rows() || errorCovariance->rows() != Q->rows() || errorCovariance->cols() != Q->cols())
        {
            return false;
        }

        MatrixXd Qdiag = Q->matrix.diagonal().asDiagonal();
        errorCovariance->matrix = F->matrix * P->matrix.selfadjointView<Eigen::Upper>() * F->matrix.transpose() + Qdiag;

        return true;

    }
}

bool NativeKalmanFilterImpl::computeKalmanGain(NativeMatrixImpl *gain, NativeMatrixImpl *P, NativeMatrixImpl *H, NativeMatrixImpl *R)
{

    if (H->cols() != P->rows() || P->rows() != P->cols() || H->rows() != R->rows() || R->cols() != 1 || gain->rows() != P->rows() || gain->cols() != R->rows())
    {
        return false;
    }

    MatrixXd PHt = P->matrix.selfadjointView<Eigen::Upper>() * H->matrix.transpose();
    MatrixXd Rdiag = R->matrix.asDiagonal();
    MatrixXd toInvert = H->matrix * PHt + Rdiag;


    gain->matrix = PHt * toInvert.inverse();


    return true;
}

bool NativeKalmanFilterImpl::updateState(NativeMatrixImpl *nextState, NativeMatrixImpl *x, NativeMatrixImpl *K, NativeMatrixImpl *r)
{
    if (x->rows() != K->rows() || r->rows() != K->cols() || x->cols() != 1 || r->cols() != 1 || nextState->rows() != x->rows() || nextState->cols() != 1)
    {
        return false;
    }

    nextState->matrix = x->matrix + K->matrix * r->matrix;

    return true;
}

bool NativeKalmanFilterImpl::updateErrorCovariance(NativeMatrixImpl *nextError, NativeMatrixImpl *K, NativeMatrixImpl *H, NativeMatrixImpl *P)
{
    if (K->cols() != H->rows() || P->rows() != P->cols() || K->rows() != H->cols() || P->rows() != H->cols() || nextError->rows() != P->rows() || nextError->cols() != P->cols())
    {
       return false;
    }

    nextError->matrix = (MatrixXd::Identity(P->rows(), P->rows()) - K->matrix * H->matrix) * P->matrix.selfadjointView<Eigen::Upper>();


    return true;
}
