/*
 * NativeCommonOps.cpp
 *
 *  Created on: Nov 27, 2018
 *      Author: Georg Wiedebach
 */

#include <jni.h>
#include <Eigen/Dense>
#include <iostream>
#include "us_ihmc_matrixlib_NativeCommonOpsWrapper.h"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> JMatrixMap;

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_mult(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows, jint aCols, jint bCols)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aCols);
   JMatrixMap B(bDataArray, aCols, bCols);
   JMatrixMap x(resultDataArray, aRows, bCols);

   x.noalias() = A * B;

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_multQuad(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows, jint aCols)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aCols);
   JMatrixMap B(bDataArray, aRows, aRows);
   JMatrixMap x(resultDataArray, aCols, aCols);

   x.noalias() = A.transpose() * B * A;

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_invert(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jint aRows)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aRows);
   JMatrixMap x(resultDataArray, aRows, aRows);

   x.noalias() = A.lu().inverse();

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_solve(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aRows);
   JMatrixMap B(bDataArray, aRows, 1);
   JMatrixMap x(resultDataArray, aRows, 1);

   x.noalias() = A.lu().solve(B);

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT jboolean JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_solveCheck(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aRows);
   JMatrixMap B(bDataArray, aRows, 1);
   JMatrixMap x(resultDataArray, aRows, 1);

   auto fullPivLu = A.fullPivLu();
   bool invertible = fullPivLu.isInvertible();
   if (invertible)
      x.noalias() = fullPivLu.solve(B);

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);

   return invertible;
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_solveRobust(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows, jint aCols)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aCols);
   JMatrixMap B(bDataArray, aRows, 1);
   JMatrixMap x(resultDataArray, aCols, 1);

   x.noalias() = A.householderQr().solve(B);

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_solveDamped(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows, jint aCols, jdouble alpha)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aCols);
   JMatrixMap B(bDataArray, aRows, 1);
   JMatrixMap x(resultDataArray, aCols, 1);

   auto outer = A * A.transpose() + MatrixXd::Identity(aRows, aRows) * alpha * alpha;
   x.noalias() = A.transpose() * outer.llt().solve(B);

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}

JNIEXPORT void JNICALL Java_us_ihmc_matrixlib_NativeCommonOpsWrapper_projectOnNullspace(JNIEnv *env, jobject thisObj,
      jdoubleArray result, jdoubleArray aData, jdoubleArray bData, jint aRows, jint aCols, jint bRows, jdouble alpha)
{
   jdouble *aDataArray = (jdouble*) env->GetPrimitiveArrayCritical(aData, NULL);
   jdouble *bDataArray = (jdouble*) env->GetPrimitiveArrayCritical(bData, NULL);
   jdouble *resultDataArray = (jdouble*) env->GetPrimitiveArrayCritical(result, NULL);

   JMatrixMap A(aDataArray, aRows, aCols);
   JMatrixMap B(bDataArray, bRows, aCols);
   JMatrixMap x(resultDataArray, aRows, aCols);

   auto BtB = B.transpose() * B;
   auto outer = BtB + MatrixXd::Identity(aCols, aCols) * alpha * alpha;
   x.noalias() = A * (MatrixXd::Identity(aCols, aCols) - outer.llt().solve(BtB));

   env->ReleasePrimitiveArrayCritical(aData, aDataArray, 0);
   env->ReleasePrimitiveArrayCritical(bData, bDataArray, 0);
   env->ReleasePrimitiveArrayCritical(result, resultDataArray, 0);
}
