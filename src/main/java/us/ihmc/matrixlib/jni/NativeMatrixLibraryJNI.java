/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package us.ihmc.matrixlib.jni;

public class NativeMatrixLibraryJNI {
  public final static native void NativeMatrixImpl_nan_set(long jarg1, NativeMatrixImpl jarg1_, double jarg2);
  public final static native double NativeMatrixImpl_nan_get(long jarg1, NativeMatrixImpl jarg1_);
  public final static native long new_NativeMatrixImpl(int jarg1, int jarg2);
  public final static native void NativeMatrixImpl_resize(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3);
  public final static native void NativeMatrixImpl_growRows(long jarg1, NativeMatrixImpl jarg1_, int jarg2);
  public final static native boolean NativeMatrixImpl_set__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_);
  public final static native boolean NativeMatrixImpl_add__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_add__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, double jarg3, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_add__SWIG_2(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, double jarg4, long jarg5, NativeMatrixImpl jarg5_);
  public final static native boolean NativeMatrixImpl_addEquals__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_);
  public final static native boolean NativeMatrixImpl_addEquals__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_add__SWIG_3(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, double jarg4);
  public final static native boolean NativeMatrixImpl_subtract(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_mult__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_mult__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_multAdd__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multAdd__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_multTransA__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multTransA__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_multAddTransA__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multAddTransA__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_multTransB__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multTransB__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_multAddTransB__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multAddTransB__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeMatrixImpl_addBlock__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8, double jarg9);
  public final static native boolean NativeMatrixImpl_addBlock__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8);
  public final static native boolean NativeMatrixImpl_subtractBlock(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8);
  public final static native boolean NativeMatrixImpl_multAddBlock__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, int jarg4, int jarg5);
  public final static native boolean NativeMatrixImpl_multAddBlock__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_, int jarg5, int jarg6);
  public final static native boolean NativeMatrixImpl_multAddBlockTransA__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, int jarg4, int jarg5);
  public final static native boolean NativeMatrixImpl_multAddBlockTransA__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_, int jarg5, int jarg6);
  public final static native boolean NativeMatrixImpl_multQuad(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multAddQuad(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_multQuadBlock(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, int jarg4, int jarg5);
  public final static native boolean NativeMatrixImpl_multAddQuadBlock(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, int jarg4, int jarg5);
  public final static native boolean NativeMatrixImpl_invert(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_);
  public final static native boolean NativeMatrixImpl_solve(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_solveCheck(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_insert__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8);
  public final static native boolean NativeMatrixImpl_insert__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double[] jarg2, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8, int jarg9, int jarg10);
  public final static native boolean NativeMatrixImpl_insert__SWIG_2(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, double jarg4, double jarg5, double jarg6, double jarg7, double jarg8, double jarg9, double jarg10, double jarg11, double jarg12);
  public final static native boolean NativeMatrixImpl_insertTupleRow(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, double jarg4, double jarg5, double jarg6);
  public final static native boolean NativeMatrixImpl_insertScaled__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8, double jarg9);
  public final static native boolean NativeMatrixImpl_insertScaled__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double[] jarg2, int jarg3, int jarg4, int jarg5, int jarg6, int jarg7, int jarg8, int jarg9, int jarg10, double jarg11);
  public final static native boolean NativeMatrixImpl_extract(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5, double[] jarg6, int jarg7, int jarg8, int jarg9, int jarg10);
  public final static native boolean NativeMatrixImpl_transpose(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_);
  public final static native boolean NativeMatrixImpl_removeRow(long jarg1, NativeMatrixImpl jarg1_, int jarg2);
  public final static native boolean NativeMatrixImpl_removeColumn(long jarg1, NativeMatrixImpl jarg1_, int jarg2);
  public final static native void NativeMatrixImpl_zero(long jarg1, NativeMatrixImpl jarg1_);
  public final static native boolean NativeMatrixImpl_containsNaN(long jarg1, NativeMatrixImpl jarg1_);
  public final static native boolean NativeMatrixImpl_scale__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, double jarg2, long jarg3, NativeMatrixImpl jarg3_);
  public final static native boolean NativeMatrixImpl_scaleBlock(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5, double jarg6);
  public final static native boolean NativeMatrixImpl_isAprrox(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, double jarg3);
  public final static native boolean NativeMatrixImpl_set__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double[] jarg2, int jarg3, int jarg4);
  public final static native boolean NativeMatrixImpl_get__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, double[] jarg2, int jarg3, int jarg4);
  public final static native boolean NativeMatrixImpl_addDiagonal__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5, double jarg6);
  public final static native boolean NativeMatrixImpl_fill(long jarg1, NativeMatrixImpl jarg1_, double jarg2);
  public final static native boolean NativeMatrixImpl_fillDiagonal__SWIG_0(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5, double jarg6);
  public final static native boolean NativeMatrixImpl_fillBlock(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5, double jarg6);
  public final static native boolean NativeMatrixImpl_setElement(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, long jarg4, NativeMatrixImpl jarg4_, int jarg5, int jarg6);
  public final static native boolean NativeMatrixImpl_zeroBlock(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, int jarg5);
  public final static native boolean NativeMatrixImpl_addDiagonal__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, double jarg5);
  public final static native boolean NativeMatrixImpl_addDiagonal__SWIG_2(long jarg1, NativeMatrixImpl jarg1_, double jarg2);
  public final static native boolean NativeMatrixImpl_fillDiagonal__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, int jarg4, double jarg5);
  public final static native boolean NativeMatrixImpl_fillDiagonal__SWIG_2(long jarg1, NativeMatrixImpl jarg1_, double jarg2);
  public final static native double NativeMatrixImpl_min(long jarg1, NativeMatrixImpl jarg1_);
  public final static native double NativeMatrixImpl_max(long jarg1, NativeMatrixImpl jarg1_);
  public final static native double NativeMatrixImpl_sum(long jarg1, NativeMatrixImpl jarg1_);
  public final static native double NativeMatrixImpl_prod(long jarg1, NativeMatrixImpl jarg1_);
  public final static native void NativeMatrixImpl_scale__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, double jarg2);
  public final static native boolean NativeMatrixImpl_set__SWIG_2(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3, double jarg4);
  public final static native double NativeMatrixImpl_get__SWIG_1(long jarg1, NativeMatrixImpl jarg1_, int jarg2, int jarg3);
  public final static native int NativeMatrixImpl_rows(long jarg1, NativeMatrixImpl jarg1_);
  public final static native int NativeMatrixImpl_cols(long jarg1, NativeMatrixImpl jarg1_);
  public final static native int NativeMatrixImpl_size(long jarg1, NativeMatrixImpl jarg1_);
  public final static native boolean NativeMatrixImpl_zeroRow(long jarg1, NativeMatrixImpl jarg1_, int jarg2);
  public final static native boolean NativeMatrixImpl_zeroCol(long jarg1, NativeMatrixImpl jarg1_, int jarg2);
  public final static native void NativeMatrixImpl_print(long jarg1, NativeMatrixImpl jarg1_);
  public final static native void delete_NativeMatrixImpl(long jarg1);
  public final static native long new_NativeNullspaceProjectorImpl(int jarg1);
  public final static native boolean NativeNullspaceProjectorImpl_projectOnNullSpace(long jarg1, NativeNullspaceProjectorImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_, double jarg5);
  public final static native void delete_NativeNullspaceProjectorImpl(long jarg1);
  public final static native long new_NativeKalmanFilterImpl();
  public final static native boolean NativeKalmanFilterImpl_predictErrorCovariance(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeKalmanFilterImpl_computeKalmanGain(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeKalmanFilterImpl_updateState(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native boolean NativeKalmanFilterImpl_updateErrorCovariance(long jarg1, NativeMatrixImpl jarg1_, long jarg2, NativeMatrixImpl jarg2_, long jarg3, NativeMatrixImpl jarg3_, long jarg4, NativeMatrixImpl jarg4_);
  public final static native void delete_NativeKalmanFilterImpl(long jarg1);
}
