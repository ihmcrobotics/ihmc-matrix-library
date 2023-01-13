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
    return NativeMatrixLibraryJNI.NativeMatrixImpl_add__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean add(NativeMatrixImpl a, double scale, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_add__SWIG_1(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, scale, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean add(double scale1, NativeMatrixImpl a, double scale2, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_add__SWIG_2(swigCPtr, this, scale1, NativeMatrixImpl.getCPtr(a), a, scale2, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean addEquals(NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addEquals__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean addEquals(double scale, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addEquals__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean add(int row, int col, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_add__SWIG_3(swigCPtr, this, row, col, value);
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
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAdd__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAdd(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAdd__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransA(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransA__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransA(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransA__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransA(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransA__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransA(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransA__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransB(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransB__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multTransB(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multTransB__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransB(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransB__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddTransB(double scale, NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddTransB__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean addBlock(NativeMatrixImpl a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns, double scale) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addBlock__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns, scale);
  }

  public boolean addBlock(NativeMatrixImpl a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addBlock__SWIG_1(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns);
  }

  public boolean subtractBlock(NativeMatrixImpl a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_subtractBlock(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns);
  }

  public boolean multAddBlock(NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddBlock__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multAddBlock(double scale, NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddBlock__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multAddBlockTransA(NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddBlockTransA__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multAddBlockTransA(double scale, NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddBlockTransA__SWIG_1(swigCPtr, this, scale, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multQuad(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multQuad(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multAddQuad(NativeMatrixImpl a, NativeMatrixImpl b) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddQuad(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b);
  }

  public boolean multQuadBlock(NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multQuadBlock(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
  }

  public boolean multAddQuadBlock(NativeMatrixImpl a, NativeMatrixImpl b, int rowStart, int colStart) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_multAddQuadBlock(swigCPtr, this, NativeMatrixImpl.getCPtr(a), a, NativeMatrixImpl.getCPtr(b), b, rowStart, colStart);
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

  public boolean insert(int startRow, int startCol, double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insert__SWIG_2(swigCPtr, this, startRow, startCol, m00, m01, m02, m10, m11, m12, m20, m21, m22);
  }

  public boolean insertTupleRow(int startRow, int startCol, double x, double y, double z) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insertTupleRow(swigCPtr, this, startRow, startCol, x, y, z);
  }

  public boolean insertScaled(NativeMatrixImpl src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0, double scale) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insertScaled__SWIG_0(swigCPtr, this, NativeMatrixImpl.getCPtr(src), src, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0, scale);
  }

  public boolean insertScaled(double[] src, int srcRows, int srcCols, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0, double scale) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_insertScaled__SWIG_1(swigCPtr, this, src, srcRows, srcCols, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0, scale);
  }

  public boolean extract(int srcY0, int srcY1, int srcX0, int srcX1, double[] dst, int dstRows, int dstCols, int dstY0, int dstX0) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_extract(swigCPtr, this, srcY0, srcY1, srcX0, srcX1, dst, dstRows, dstCols, dstY0, dstX0);
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

  public boolean scaleBlock(int startRow, int startCol, int numberOfRows, int numberOfCols, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_scaleBlock(swigCPtr, this, startRow, startCol, numberOfRows, numberOfCols, value);
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

  public boolean addDiagonal(int startRow, int startCol, int rows, int cols, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addDiagonal__SWIG_0(swigCPtr, this, startRow, startCol, rows, cols, value);
  }

  public boolean fill(double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_fill(swigCPtr, this, value);
  }

  public boolean fillDiagonal(int startRow, int startCol, int rows, int cols, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_fillDiagonal__SWIG_0(swigCPtr, this, startRow, startCol, rows, cols, value);
  }

  public boolean fillBlock(int startRow, int startCol, int numberOfRows, int numberOfCols, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_fillBlock(swigCPtr, this, startRow, startCol, numberOfRows, numberOfCols, value);
  }

  public boolean setElement(int dstRow, int dstCol, NativeMatrixImpl src, int srcRow, int srcCol) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_setElement(swigCPtr, this, dstRow, dstCol, NativeMatrixImpl.getCPtr(src), src, srcRow, srcCol);
  }

  public boolean zeroBlock(int srcY0, int srcY1, int srcX0, int srcX1) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_zeroBlock(swigCPtr, this, srcY0, srcY1, srcX0, srcX1);
  }

  public boolean addDiagonal(int startRow, int startCol, int size, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addDiagonal__SWIG_1(swigCPtr, this, startRow, startCol, size, value);
  }

  public boolean addDiagonal(double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_addDiagonal__SWIG_2(swigCPtr, this, value);
  }

  public boolean fillDiagonal(int startRow, int startCol, int size, double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_fillDiagonal__SWIG_1(swigCPtr, this, startRow, startCol, size, value);
  }

  public boolean fillDiagonal(double value) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_fillDiagonal__SWIG_2(swigCPtr, this, value);
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

  public boolean zeroRow(int rowToZero) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_zeroRow(swigCPtr, this, rowToZero);
  }

  public boolean zeroCol(int colToZero) {
    return NativeMatrixLibraryJNI.NativeMatrixImpl_zeroCol(swigCPtr, this, colToZero);
  }

  public void print() {
    NativeMatrixLibraryJNI.NativeMatrixImpl_print(swigCPtr, this);
  }

}
