/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package us.ihmc.matrixlib.jni;

public class NativeMatrixImpl {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected NativeMatrixImpl(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(NativeMatrixImpl obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        NativeMatrixLibraryJNI.delete_NativeMatrixImpl(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setNan(double value) {
    NativeMatrixLibraryJNI.NativeMatrixImpl_nan_set(swigCPtr, this, value);
  }

  public double getNan() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_nan_get(swigCPtr, this);
  }

  public NativeMatrixImpl(int numRows, int numCols) {
    this(NativeMatrixLibraryJNI.new_NativeMatrixImpl(numRows, numCols), true);
  }

  public void resize(int numRows, int numCols) {
    NativeMatrixLibraryJNI.NativeMatrixImpl_resize(swigCPtr, this, numRows, numCols);
  }

  public boolean set(NativeMatrixImpl a) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_set__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a);
  }

  public boolean add(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_add(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean subtract(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_subtract(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean mult(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_mult__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean mult(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_mult__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAdd(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAdd(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransA(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransA(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransA(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransA(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransB(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransB(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransB(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransB(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean addBlock(NativeMatrixImpl a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns, double scale) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addBlock(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns, scale);
  }

  public boolean multAddBlock(NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddBlock(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multQuad(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multQuad(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean invert(NativeMatrixImpl a) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_invert(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a);
  }

  public boolean solve(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_solve(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean solveCheck(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_solveCheck(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean insert(NativeMatrixImpl src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insert__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(src), src, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0);
  }

  public boolean insert(double[] src, int rows, int cols, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insert__SWIG_1(swigCPtr, this, src, rows, cols, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0);
  }

  public boolean extract(int srcY0, int srcY1, int srcX0, int srcX1, double[] dst, int dstRows, int dstCols, int dstY0, int dstX0) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_extract(swigCPtr, this, srcY0, srcY1, srcX0, srcX1, dst, dstRows, dstCols, dstY0, dstX0);
  }

  public boolean projectOnNullSpace(NativeMatrixImpl A, NativeMatrixImpl B, double alpha) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_projectOnNullSpace(swigCPtr, this, NativeMatrixImpl.getCPtr(A), A, NativeMatrixImpl.getCPtr(B), B, alpha);
  }

  public boolean transpose(NativeMatrixImpl a) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_transpose(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a);
  }

  public boolean removeRow(int indexToRemove) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_removeRow(swigCPtr, this, indexToRemove);
  }

  public boolean removeColumn(int indexToRemove) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_removeColumn(swigCPtr, this, indexToRemove);
  }

  public void zero() {
    NativeMatrixLibraryJNI.NativeMatrixImpl_zero(swigCPtr, this);
  }

  public boolean containsNaN() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_containsNaN(swigCPtr, this);
  }

  public boolean scale(double scale, NativeMatrixImpl src) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_scale__SWIG_0(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(src), src);
  }

  public boolean isAprrox(NativeMatrixImpl other, double precision) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_isAprrox(swigCPtr, this, NativeMatrixImpl.getCPtr(other), other, precision);
  }

  public boolean set(double[] data, int rows, int cols) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_set__SWIG_1(swigCPtr, this, data, rows, cols);
  }

  public boolean get(double[] data, int rows, int cols) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_get__SWIG_0(swigCPtr, this, data, rows, cols);
  }

  public double min() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_min(swigCPtr, this);
  }

  public double max() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_max(swigCPtr, this);
  }

  public double sum() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_sum(swigCPtr, this);
  }

  public double prod() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_prod(swigCPtr, this);
  }

  public void scale(double scale) {
    NativeMatrixLibraryJNI.NativeMatrixImpl_scale__SWIG_1(swigCPtr, this, scale);
  }

  public boolean set(int row, int col, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_set__SWIG_2(swigCPtr, this, row, col, value);
  }

  public double get(int row, int col) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_get__SWIG_1(swigCPtr, this, row, col);
  }

  public int rows() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_rows(swigCPtr, this);
  }

  public int cols() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_cols(swigCPtr, this);
  }

  public int size() {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_size(swigCPtr, this);
  }

  public void print() {
    NativeMatrixLibraryJNI.NativeMatrixImpl_print(swigCPtr, this);
  }

}
