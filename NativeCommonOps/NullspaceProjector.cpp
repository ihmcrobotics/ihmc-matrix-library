#include "NullspaceProjector.h"
#include <iostream>

NullspaceProjectorImpl::NullspaceProjectorImpl(int degreesOfFreedom) :
    degreesOfFreedom_(degreesOfFreedom),
    identity(Eigen::MatrixXd::Identity(degreesOfFreedom, degreesOfFreedom)),
    BtB(degreesOfFreedom, degreesOfFreedom),
    outer(degreesOfFreedom, degreesOfFreedom)
{

}



bool NullspaceProjectorImpl::projectOnNullSpace(NativeMatrixImpl* A, NativeMatrixImpl* B, NativeMatrixImpl* x, double alpha)
{


    if(B->cols() != degreesOfFreedom_)
    {
        return false;
    }

    int aCols = A->cols();

    if(aCols != degreesOfFreedom_)
    {
        return false;
    }

    BtB = B->matrix.transpose() * B->matrix;
    outer = BtB + identity * alpha * alpha;

//    std::cout << BtB << std::endl;
//    std::cout << outer << std::endl;
    x->resize(A->rows(), aCols);

    x->matrix = A->matrix * (identity - outer.llt().solve(BtB));


    return true;
}
