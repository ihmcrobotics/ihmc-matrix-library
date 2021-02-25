/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package us.ihmc.matrixlib.jni;

public class NativeKalmanFilterImpl {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected NativeKalmanFilterImpl(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(NativeKalmanFilterImpl obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        NativeMatrixLibraryJNI.delete_NativeKalmanFilterImpl(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public NativeKalmanFilterImpl() {
    this(NativeMatrixLibraryJNI.new_NativeKalmanFilterImpl(), true);
  }

  public static boolean predictErrorCovariance(NativeMatrixImpl errorCovariance, NativeMatrixImpl F, NativeMatrixImpl P, NativeMatrixImpl Q) {
    return NativeMatrixLibraryJNI.NativeKalmanFilterImpl_predictErrorCovariance(NativeMatrixImpl.getCPtr(errorCovariance), errorCovariance, NativeMatrixImpl.getCPtr(F), F, NativeMatrixImpl.getCPtr(P), P, NativeMatrixImpl.getCPtr(Q), Q);
  }

  public static boolean computeKalmanGain(NativeMatrixImpl gain, NativeMatrixImpl P, NativeMatrixImpl H, NativeMatrixImpl R) {
    return NativeMatrixLibraryJNI.NativeKalmanFilterImpl_computeKalmanGain(NativeMatrixImpl.getCPtr(gain), gain, NativeMatrixImpl.getCPtr(P), P, NativeMatrixImpl.getCPtr(H), H, NativeMatrixImpl.getCPtr(R), R);
  }

  public static boolean updateState(NativeMatrixImpl nextState, NativeMatrixImpl x, NativeMatrixImpl K, NativeMatrixImpl r) {
    return NativeMatrixLibraryJNI.NativeKalmanFilterImpl_updateState(NativeMatrixImpl.getCPtr(nextState), nextState, NativeMatrixImpl.getCPtr(x), x, NativeMatrixImpl.getCPtr(K), K, NativeMatrixImpl.getCPtr(r), r);
  }

  public static boolean updateErrorCovariance(NativeMatrixImpl nextError, NativeMatrixImpl K, NativeMatrixImpl H, NativeMatrixImpl P) {
    return NativeMatrixLibraryJNI.NativeKalmanFilterImpl_updateErrorCovariance(NativeMatrixImpl.getCPtr(nextError), nextError, NativeMatrixImpl.getCPtr(K), K, NativeMatrixImpl.getCPtr(H), H, NativeMatrixImpl.getCPtr(P), P);
  }

}
