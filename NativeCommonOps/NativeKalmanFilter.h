#ifndef NATIVEKALMANFILTER_H
#define NATIVEKALMANFILTER_H


#include <Eigen/Dense>
#include "NativeMatrix.h"

class NativeKalmanFilterImpl
{
public:
    NativeKalmanFilterImpl();



static bool predictErrorCovariance(NativeMatrixImpl *errorCovariance, NativeMatrixImpl *F, NativeMatrixImpl *P, NativeMatrixImpl *Q);

static bool computeKalmanGain(NativeMatrixImpl *gain, NativeMatrixImpl *P, NativeMatrixImpl *H, NativeMatrixImpl *R);

static bool updateState(NativeMatrixImpl *nextState, NativeMatrixImpl *x, NativeMatrixImpl *K , NativeMatrixImpl *r);

static bool updateErrorCovariance(NativeMatrixImpl *nextError, NativeMatrixImpl *K, NativeMatrixImpl *H, NativeMatrixImpl *P);

};

#endif // NATIVEKALMANFILTER_H
