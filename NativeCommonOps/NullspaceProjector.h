#ifndef NULLSPACEPROJECTOR_H
#define NULLSPACEPROJECTOR_H

#include <Eigen/Dense>
#include "NativeMatrix.h"

class NullspaceProjectorImpl
{
public:
    NullspaceProjectorImpl(int degreesOfFreedom);
    bool projectOnNullSpace(NativeMatrixImpl *A, NativeMatrixImpl *B, NativeMatrixImpl *x, double alpha);


private:
    int degreesOfFreedom_;
    Eigen::MatrixXd identity;
    Eigen::MatrixXd BtB;
    Eigen::MatrixXd outer;
};

#endif // NULLSPACEPROJECTOR_H
