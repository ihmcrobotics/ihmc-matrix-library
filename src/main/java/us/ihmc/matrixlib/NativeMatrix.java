package us.ihmc.matrixlib;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.Matrix;
import org.ejml.data.MatrixType;
import org.ejml.data.ReshapeMatrix;
import org.ejml.ops.MatrixIO;

import us.ihmc.euclid.matrix.interfaces.Matrix3DReadOnly;
import us.ihmc.euclid.tuple3D.interfaces.Tuple3DReadOnly;
import us.ihmc.matrixlib.jni.NativeMatrixImpl;
import us.ihmc.tools.nativelibraries.NativeLibraryLoader;

/**
 * {@code NativeMatrix} is dense matrix with real elements that are 64-bits floats. Unlike
 * {@link DMatrixRMaj}, the operations on {@code NativeMatrix} are executed in C++ using Eigen
 * leading to significant performance improvement when compared to the regular Java implementation.
 * <p>
 * Note that unlike {@link DMatrixRMaj}, the matrix is stored internally in a column-major 1D array
 * format, for example:<br>
 * data =
 * </p>
 * 
 * <pre>
 * a[0]  a[4]  a[8]   a[12]
 * a[1]  a[5]  a[9]   a[13]
 * a[2]  a[6]  a[10]  a[14]
 * a[3]  a[7]  a[11]  a[15]
 * </pre>
 * </p>
 *
 * @author Jesper Smith
 */
public class NativeMatrix implements ReshapeMatrix, DMatrix
{
   private static final long serialVersionUID = -6143897236850269840L;

   static
   {
      NativeLibraryLoader.loadLibrary("", "NativeCommonOps");
   }

   final NativeMatrixImpl impl;

   /**
    * Creates a new matrix with the specified shape whose elements initially have the value of zero.
    *
    * @param rows The number of rows in the matrix.
    * @param cols The number of columns in the matrix.
    */
   public NativeMatrix(int rows, int cols)
   {
      impl = new NativeMatrixImpl(rows, cols);
      zero();
   }

   /**
    * Creates a new matrix which is equivalent to the provided matrix.
    *
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public NativeMatrix(DMatrixRMaj matrix)
   {
      this(matrix.getNumRows(), matrix.getNumCols());
      set(matrix);
   }

   /**
    * Creates a new matrix which is equivalent to the provided matrix.
    *
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public NativeMatrix(NativeMatrix matrix)
   {
      this(matrix.getNumRows(), matrix.getNumCols());
      set(matrix);
   }

   /**
    * Changes the number of rows and columns in the matrix, allowing its size to grow or shrink.
    * <p>
    * The primary use for this function is to encourage data reuse and avoid unnecessarily declaring
    * and initialization of new memory.
    * </p>
    * <p>
    * Examples:<br>
    * [ 1 2 ; 3 4 ] &rarr; reshape( 1 , 2 ) = [ 1 2 ]<br>
    * [ 1 2 ; 3 4 ] &rarr; reshape( 2 , 1 ) = [ 1 ; 2 ]<br>
    * [ 1 2 ; 3 4 ] &rarr; reshape( 2 , 3 ) = [ 0 0 0 ; 0 0 0 ]
    * </p>
    *
    * @param rows The new number of rows in the matrix.
    * @param cols The new number of columns in the matrix.
    */
   @Override
   public void reshape(int rows, int cols)
   {
      reshape(rows, cols, true);
   }

   public void reshape(int rows, int cols, boolean keepValues)
   {
      impl.resize(rows, cols);

      if (!keepValues)
         impl.zero();
   }

   /**
    * Copies the given matrix and scales every single element by the given factor.
    * <p>
    * This operation reshapes this to match the given matrix.
    * </p>
    *
    * @param alpha  the scale factor to apply to every element.
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public void scale(double alpha, DMatrixRMaj matrix)
   {
      set(matrix);
      scale(alpha);
   }

   /**
    * Copies the given matrix and scales every single element by the given factor.
    * <p>
    * This operation reshapes this to match the given matrix.
    * </p>
    *
    * @param alpha  the scale factor to apply to every element.
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public void scale(double alpha, NativeMatrix matrix)
   {
      if (!impl.scale(alpha, matrix.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Copies the given matrix into this.
    * <p>
    * This operation reshapes this to match the given matrix.
    * </p>
    *
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public void set(DMatrixRMaj matrix)
   {
      if (!impl.set(matrix.data, matrix.numRows, matrix.numCols))
      {
         throw new IllegalArgumentException("Cannot set matrix.");
      }
   }

   /**
    * Copies the given matrix into this.
    * <p>
    * This operation reshapes this to match the given matrix.
    * </p>
    *
    * @param matrix The matrix which is to be copied. This is not modified or saved.
    */
   public void set(NativeMatrix matrix)
   {
      if (!impl.set(matrix.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Packs this matrix into a {@code DMatrixRMaj}.
    *
    * @param matrixToPack the matrix used to store this. Modified.
    */
   public void get(DMatrixRMaj matrixToPack)
   {
      matrixToPack.reshape(getNumRows(), getNumCols());

      if (!impl.get(matrixToPack.data, matrixToPack.numRows, matrixToPack.numCols))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix addition</br>
    * this = a + b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a left matrix in addition. Not modified.
    * @param b right matrix in addition. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void add(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.add(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Adds value to an element in the matrix
    *
    * @param row row access to data
    * @param col col access to data
    * @param value value to add
    * @throws IllegalArgumentException if the accessors are out of bounds.
    */
   public void add(int row, int col, double value)
   {
      if (impl.add(row, col, value))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix subtraction</br>
    * this = a - b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a left matrix in subtraction. Not modified.
    * @param b right matrix in subtraction. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void subtract(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.subtract(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = a * b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a left matrix in multiplication. Not modified.
    * @param b right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void mult(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.mult(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = scale * a * b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a     left matrix in multiplication. Not modified.
    * @param b     right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void mult(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.mult(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += a * b
    *
    * @param a matrix in multiplication. Not modified.
    * @param b matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAdd(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAdd(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += scale * a * b
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a     matrix in multiplication. Not modified.
    * @param b     matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAdd(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAdd(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += a<sup>T</sup> * b
    *
    * @param a left matrix in multiplication. Not modified.
    * @param b right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddTransA(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAddTransA(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += scale * a<sup>T</sup> * b
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a left matrix in multiplication. Not modified.
    * @param b right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddTransA(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAddTransA(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += a * b<sup>T</sup>
    *
    * @param a left matrix in multiplication. Not modified.
    * @param b right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddTransB(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAddTransB(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Add the result of the matrix multiplication to this<br>
    * this += scale * a * b<sup>T</sup>
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a left matrix in multiplication. Not modified.
    * @param b right matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddTransB(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAddTransB(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Performs the following operation:<br>
    * this += a * b <br>
    * where we are only modifying a block of this matrix, starting a rowStart, colStart
    *
    * @param a        The left matrix in the multiplication operation. Not modified.
    * @param b        The right matrix in the multiplication operation. Not modified.
    * @param rowStart first row index of the block to process.
    * @param colStart first column index of the block to process.
    */
   public void multAddBlock(NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      if (!impl.multAddBlock(a.impl, b.impl, rowStart, colStart))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Performs the following operation:<br>
    * this += scale * a * b <br>
    * where we are only modifying a block of this matrix, starting a rowStart, colStart
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a        The left matrix in the multiplication operation. Not modified.
    * @param b        The right matrix in the multiplication operation. Not modified.
    * @param rowStart first row index of the block to process.
    * @param colStart first column index of the block to process.
    */
   public void multAddBlock(double scale, NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      if (!impl.multAddBlock(scale, a.impl, b.impl, rowStart, colStart))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Performs the following operation:<br>
    * this += scale * a <br>
    * where only a block of the matrix a is scaled then added to a block of same size in this.
    *
    * @param a               The matrix to add to this. Not modified.
    * @param destStartRow    The first row index of the block in this.
    * @param destStartColumn The first column index of the block in this.
    * @param srcStartRow     The first row index of the block in the matrix a.
    * @param srcStartColumn  The first column index of the block in matrix a.
    * @param numberOfRows    The number of rows of the block.
    * @param numberOfColumns The number of columns of the block.
    */
   public void addBlock(NativeMatrix a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns,
                        double scale)
   {
      if (!impl.addBlock(a.impl, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns, scale))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Performs the following operation:<br>
    * this += a <br>
    * where only a block of the matrix a is added to a block of same size in this.
    *
    * @param a               The matrix to add to this. Not modified.
    * @param destStartRow    The first row index of the block in this.
    * @param destStartColumn The first column index of the block in this.
    * @param srcStartRow     The first row index of the block in the matrix a.
    * @param srcStartColumn  The first column index of the block in matrix a.
    * @param numberOfRows    The number of rows of the block.
    * @param numberOfColumns The number of columns of the block.
    */
   public void addBlock(NativeMatrix a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns)
   {
      if (!impl.addBlock(a.impl, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Performs the following operation:<br>
    * this -= a <br>
    * where only a block of the matrix a is subtracted from a block of same size in this.
    *
    * @param a               The matrix to subtract from this. Not modified.
    * @param destStartRow    The first row index of the block in this.
    * @param destStartColumn The first column index of the block in this.
    * @param srcStartRow     The first row index of the block in the matrix a.
    * @param srcStartColumn  The first column index of the block in matrix a.
    * @param numberOfRows    The number of rows of the block.
    * @param numberOfColumns The number of columns of the block.
    */
   public void subtractBlock(NativeMatrix a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns)
   {
      if (!impl.subtractBlock(a.impl, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = a * b<sup>T</sup>
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a The left matrix in the multiplication operation. Not modified.
    * @param b The right matrix in the multiplication operation. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multTransB(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multTransB(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = scale * a * b<sup>T</sup>
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a The left matrix in the multiplication operation. Not modified.
    * @param b The right matrix in the multiplication operation. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multTransB(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multTransB(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = a<sup>T</sup> * b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a The left matrix in the multiplication operation. Not modified.
    * @param b The right matrix in the multiplication operation. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multTransA(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multTransA(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = scale * a<sup>T</sup> * b
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param scale the scaling factor to apply to every element of the multiplication result.
    * @param a The left matrix in the multiplication operation. Not modified.
    * @param b The right matrix in the multiplication operation. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multTransA(double scale, NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multTransA(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the quadratic form</br>
    * this = a<sup>T</sup> * b * a
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a matrix in multiplication. Not modified.
    * @param b matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multQuad(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multQuad(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the quadratic form</br>
    * this += a<sup>T</sup> * b * a
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a matrix in multiplication. Not modified.
    * @param b matrix in multiplication. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddQuad(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.multAddQuad(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the quadratic form</br>
    * this = a<sup>T</sup> * b * a
    * where only the square block product is added to a block of same size in this.
    *
    * @param a matrix in multiplication. Not modified.
    * @param b matrix in multiplication. Not modified.
    * @param rowStart first row index of the block to process.
    * @param colStart first column index of the block to process.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multQuadBlock(NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      if (!impl.multQuadBlock(a.impl, b.impl, rowStart, colStart))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the quadratic form</br>
    * this += a<sup>T</sup> * b * a
    * where only the square block product is added to a block of same size in this.
    *
    * @param a matrix in multiplication. Not modified.
    * @param b matrix in multiplication. Not modified.
    * @param rowStart first row index of the block to process.
    * @param colStart first column index of the block to process.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void multAddQuadBlock(NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      if (!impl.multAddQuadBlock(a.impl, b.impl, rowStart, colStart))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }


   /**
    * Inverts a matrix and stores the result in this.</br>
    * This method requires that the matrix is square and invertible and uses a LU decomposition.
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a matrix to invert. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void invert(NativeMatrix a)
   {
      if (a == this)
      {
         throw new IllegalArgumentException("Can not invert in place. The result matrix needs to be different from the matrix to invert.");
      }

      if (!impl.invert(a.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Transposes a matrix and stores the result in this.
    * <p>
    * This operation reshapes this to match the result of the operation.
    * </p>
    *
    * @param a the matrix to transpose. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void transpose(NativeMatrix a)
   {
      if (a == this)
      {
         throw new IllegalArgumentException("Can not transpose in place. The result matrix needs to be different from the matrix to transpose.");
      }

      if (!impl.transpose(a.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * this == b</br>
    * This method requires that the matrix a is square and invertible and uses a LU decomposition.
    *
    * @param a matrix in equation. Not modified.
    * @param b matrix in equation. Not modified.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void solve(NativeMatrix a, NativeMatrix b)
   {
      if (!impl.solve(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * this == b</br>
    * This method requires that the matrix a is square and invertible and uses a LU decomposition. This
    * method will check the invertability of the matrix a and return false if it is not invertible.
    *
    * @param a matrix in equation. Not modified.
    * @param b matrix in equation. Not modified.
    * @return whether a solution was found.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public boolean solveCheck(NativeMatrix a, NativeMatrix b)
   {
      return impl.solveCheck(a.impl, b.impl);
   }

   /**
    * Insert a matrix 3D at (startRow, startcol) in this matrix
    * 
    * @param src
    * @param startRow
    * @param startCol
    */
   public void insert(Matrix3DReadOnly src, int startRow, int startCol)
   {
      if(!impl.insert(startRow, startCol, src.getM00(), src.getM01(), src.getM02(), src.getM10(), src.getM11(), src.getM12(), src.getM20(), src.getM21(), src.getM22()))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Insert a matrix 3D at (startRow, startcol) in this matrix, scaled by scale
    * 
    * @param src
    * @param startRow
    * @param startCol
    * @param scale
    */
   public void insertScaled(Matrix3DReadOnly src, int startRow, int startCol, double scale)
   {
      if(!impl.insert(startRow, startCol, scale * src.getM00(), scale * src.getM01(), scale * src.getM02(), scale * src.getM10(), scale * src.getM11(), scale * src.getM12(), scale * src.getM20(), scale * src.getM21(), scale * src.getM22()))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Insert a tuple at (startRow, startcol) in this matrix as a row
    * 
    * @param startRow
    * @param startCol
    * @param x
    * @param y
    * @param z
    */
   public void insertTupleRow(int startRow, int startCol, double x, double y, double z)
   {
      if(!impl.insertTupleRow(startRow, startCol, x, y, z))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Insert a tuple at (startRow, startcol) in this matrix as a row
    * 
    * @param startRow
    * @param startCol
    * @param tuple
    */
   public void insertTupleRow(Tuple3DReadOnly tuple, int startRow, int startCol)
   {
      insertTupleRow(startRow, startCol, tuple.getX(), tuple.getY(), tuple.getZ());
   }
   
   /**
    * Inserts a block from the given matrix into this.
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param srcY0 The first row index (inclusive) of the block in {@code src} to copy.
    * @param srcY1 The last row index (exclusive) of the block in {@code src} to copy.
    * @param srcX0 The first column index (inclusive) of the block in {@code src} to copy.
    * @param srcX1 The last column index (exclusive) of the block in {@code src} to copy.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insert(NativeMatrix src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
   {
      if (!impl.insert(src.impl, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Inserts the given matrix into this.
    * <p>
    * {@code src} has to be either same size or smaller than this.
    * </p>
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insert(NativeMatrix src, int dstY0, int dstX0)
   {
      insert(src, 0, src.getNumRows(), 0, src.getNumCols(), dstY0, dstX0);
   }
   
   /**
    * Inserts a scaled block from the given matrix into this.
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param srcY0 The first row index (inclusive) of the block in {@code src} to copy.
    * @param srcY1 The last row index (exclusive) of the block in {@code src} to copy.
    * @param srcX0 The first column index (inclusive) of the block in {@code src} to copy.
    * @param srcX1 The last column index (exclusive) of the block in {@code src} to copy.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insertScaled(NativeMatrix src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0, double scale)
   {
      if (!impl.insertScaled(src.impl, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0, scale))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Inserts the given matrix scaled into this.
    * <p>
    * {@code src} has to be either same size or smaller than this.
    * </p>
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insertScaled(NativeMatrix src, int dstY0, int dstX0, double scale)
   {
      insertScaled(src, 0, src.getNumRows(), 0, src.getNumCols(), dstY0, dstX0, scale);
   }

   /**
    * Inserts a block from the given matrix into this.
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param srcY0 The first row index (inclusive) of the block in {@code src} to copy.
    * @param srcY1 The last row index (exclusive) of the block in {@code src} to copy.
    * @param srcX0 The first column index (inclusive) of the block in {@code src} to copy.
    * @param srcX1 The last column index (exclusive) of the block in {@code src} to copy.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insert(DMatrixRMaj src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
   {
      if (!impl.insert(src.data, src.numRows, src.numCols, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Inserts the given matrix into this.
    * <p>
    * {@code src} has to be either same size or smaller than this.
    * </p>
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insert(DMatrixRMaj src, int dstY0, int dstX0)
   {
      insert(src, 0, src.getNumRows(), 0, src.getNumCols(), dstY0, dstX0);
   }

   /**
    * Inserts a scaled block from the given matrix into this.
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param srcY0 The first row index (inclusive) of the block in {@code src} to copy.
    * @param srcY1 The last row index (exclusive) of the block in {@code src} to copy.
    * @param srcX0 The first column index (inclusive) of the block in {@code src} to copy.
    * @param srcX1 The last column index (exclusive) of the block in {@code src} to copy.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insertScaled(DMatrixRMaj src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0, double scale)
   {
      if (!impl.insertScaled(src.data, src.numRows, src.numCols, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0, scale))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   /**
    * Inserts the given matrix into this.
    * <p>
    * {@code src} has to be either same size or smaller than this.
    * </p>
    *
    * @param src   The matrix to be copied in this. Not modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void insertScaled(DMatrixRMaj src, int dstY0, int dstX0, double scale)
   {
      insertScaled(src, 0, src.getNumRows(), 0, src.getNumCols(), dstY0, dstX0, scale);
   }

   /**
    * Extracts a block from this and insert it into the given matrix.
    *
    * @param srcY0 The first row index (inclusive) of the block in {@code this} to copy.
    * @param srcY1 The last row index (exclusive) of the block in {@code this} to copy.
    * @param srcX0 The first column index (inclusive) of the block in {@code this} to copy.
    * @param srcX1 The last column index (exclusive) of the block in {@code this} to copy.
    * @param dst   The matrix in which the block is to be written. Modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code dst} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code dst} to write in.
    */
   public void extract(int srcY0, int srcY1, int srcX0, int srcX1, DMatrixRMaj dst, int dstY0, int dstX0)
   {
      if (!impl.extract(srcY0, srcY1, srcX0, srcX1, dst.data, dst.numRows, dst.numCols, dstY0, dstX0))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions");
      }
   }

   /**
    * Inserts this in the given matrix.
    * <p>
    * {@code dst} has to be either same size or larger than this.
    * </p>
    *
    * @param dst   The matrix in which the block is to be written. Modified.
    * @param dstY0 The first row index (inclusive) of the block in {@code this} to write in.
    * @param dstX0 The first column index (inclusive) of the block in {@code this} to write in.
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void extract(DMatrixRMaj dst, int dstY0, int dstX0)
   {
      extract(0, getNumRows(), 0, getNumCols(), dst, dstY0, dstX0);
   }

   /**
    * Returns the value of the specified matrix element.
    * <p>
    * IMPORTANT: Consider the overhead due to going through the JNI layer. Consider using
    * {@link #get(DMatrixRMaj)} to pack once the data back into Java land if needing to do multiple
    * accesses.
    * </p>
    *
    * @param row The row of the element.
    * @param col The column of the element.
    * @return The value of the element.
    * @throws IllegalArgumentException if either index is out of bound.
    */
   @Override
   public double get(int row, int col)
   {
      if (row < 0 || col < 0)
      {
         throwIndexOutOfBoundsException(row, col);
      }

      double value = impl.get(row, col);

      // When the index is out-of-bounds, the native layer will return NaN.
      // By performing this second check only if the result is NaN, we can reduce the overhead due to getNumRows() and getNumCols().
      if (Double.isNaN(value) && (row >= getNumRows() || col >= getNumCols()))
      {
         throwIndexOutOfBoundsException(row, col);
      }

      return value;
   }

   private void throwIndexOutOfBoundsException(int row, int col)
   {
      throw new IllegalArgumentException("Index out of bounds. Requested (" + row + ", " + col + "). Dimension (" + getNumRows() + ", " + getNumCols() + ").");
   }

   /**
    * Assigns the element in the Matrix to the specified value. <br>
    * a<sub>ij</sub> = value<br>
    * <p>
    * IMPORTANT: Consider the overhead due to going through the JNI layer. Consider using
    * {@link #get(DMatrixRMaj)} to pack once the data back into Java land if needing to do multiple
    * accesses.
    * </p>
    *
    * @param row   The row of the element.
    * @param col   The column of the element.
    * @param value The element's new value.
    * @throws IllegalArgumentException if either index is out of bound.
    */
   @Override
   public void set(int row, int col, double value)
   {
      if (!impl.set(row, col, value))
      {
         throwIndexOutOfBoundsException(row, col);
      }
   }

   /**
    * Removes a row from this and shifts the subsequent rows up by one.
    *
    * @param row the index of the row to remove.
    */
   public void removeRow(int row)
   {
      if (!impl.removeRow(row))
      {
         throw new IllegalArgumentException("Row out of bounds.");
      }
   }

   /**
    * Removes a column from this and shifts the subsequent columns left by one.
    *
    * @param col the index of the column to remove.
    */
   public void removeColumn(int col)
   {
      if (!impl.removeColumn(col))
      {
         throw new IllegalArgumentException("Col out of bounds.");
      }
   }

   /**
    * Sets all elements equal to zero.
    */
   @Override
   public void zero()
   {
      impl.zero();
   }

   /**
    * Tests whether at least one element in this is {@link Double#NaN}.
    *
    * @return {@code true} if at least one element is {@link Double#NaN}, {@code false} otherwise.
    */
   public boolean containsNaN()
   {
      return impl.containsNaN();
   }

   /**
    * Returns the number of rows in this matrix.
    *
    * @return Number of rows.
    */
   @Override
   public int getNumRows()
   {
      return impl.rows();
   }

   /**
    * Returns the number of columns in this matrix.
    *
    * @return Number of columns.
    */
   @Override
   public int getNumCols()
   {
      return impl.cols();
   }

   /**
    * Finds and returns the minimum value this matrix contains.
    *
    * @return the smallest value contained in this matrix.
    */
   public double min()
   {
      return impl.min();
   }

   /**
    * Finds and returns the maximum value this matrix contains.
    *
    * @return the greatest value contained in this matrix.
    */
   public double max()
   {
      return impl.max();
   }

   /**
    * Sums all the elements of this matrix and returns the result.
    *
    * @return the sum of all this matrix elements.
    */
   public double sum()
   {
      return impl.sum();
   }

   /**
    * Computes the product of all the elements in this matrix and returns the result.
    * 
    * @return the product of all this matrix elements.
    */
   public double prod()
   {
      return impl.prod();
   }

   /**
    * Returns the number of elements in this matrix, which is equal to the number of rows times the
    * number of columns.
    *
    * @return The number of elements in the matrix.
    */
   @Override
   public int getNumElements()
   {
      return impl.size();
   }

   /**
    * Scales the elements in this matrix by the given scale factor.
    *
    * @param scale the factor to apply to every element.
    */
   public void scale(double scale)
   {
      impl.scale(scale);
   }
   
   /**
    * Fill the diagonal of a block of the matrix with a constant value
    * 
    * @param startRow Start row for block
    * @param startCol Start col for block
    * @param size Number of elements on the diagonal to set
    * @param value Value to fill the diagonal with
    */
   public void fillDiagonal(int startRow, int startCol, int size, double value)
   {
      if(!impl.fillDiagonal(startRow, startCol, size, value))
      {
         throw new RuntimeException("Invalid matrix dimensions");
      }
   }
   
   
   /**
    * Fill a block of the matrix to a constant value
    * 
    * @param startRow Start row for block
    * @param startCol Start col for block
    * @param numberOfRows Number of rows to fill
    * @param numberOfCols Numbers of columns to fill
    * @param value Value to fill the block with
    */
   public void fillBlock(int startRow, int startCol, int numberOfRows, int numberOfCols, double value)
   {
      if(!impl.fillBlock(startRow, startCol, numberOfRows, numberOfCols, value))
      {
         throw new RuntimeException("Invalid matrix dimensions");
      }
   }
   

   /**
    * Check if the elements of this matrix are within +- "precision" to the corresponding elements in the other matrix and numRows and numCols are equal
    * 
    * @param other Matrix to check against
    * @param precision Maximum difference 
    * @return True if all elements in other are within +- "precision" to the corresponding current matrix
    */
   public boolean isApprox(NativeMatrix other, double precision)
   {
      return impl.isAprrox(other.impl, precision);
   }

   /**
    * Converts the array into a string format for display purposes. The conversion is done using
    * {@link MatrixIO#print(java.io.PrintStream, DMatrix)}.
    *
    * @return String representation of the matrix.
    */
   @Override
   public String toString()
   {
      ByteArrayOutputStream stream = new ByteArrayOutputStream();
      MatrixIO.print(new PrintStream(stream), this);
      return stream.toString();
   }

   // -------- Implementation of DMatrix API ----------------------

   @Override
   public void print()
   {
      MatrixIO.printFancy(System.out, this, MatrixIO.DEFAULT_LENGTH);
   }

   @Override
   public void print(String format)
   {
      MatrixIO.print(System.out, this, format);
   }

   @SuppressWarnings("unchecked")
   @Override
   public NativeMatrix copy()
   {
      return new NativeMatrix(this);
   }

   @SuppressWarnings("unchecked")
   @Override
   public NativeMatrix createLike()
   {
      return new NativeMatrix(getNumRows(), getNumCols());
   }

   @SuppressWarnings("unchecked")
   @Override
   public NativeMatrix create(int numRows, int numCols)
   {
      return new NativeMatrix(numRows, numCols);
   }

   /**
    * {@inheritDoc}
    * <p>
    * This implementation only supports {@link NativeMatrix} and {@link DMatrixRMaj}.
    * </p>
    * 
    * @param original The matrix which is to be copied. This is not modified or saved.
    * @throws NullPointerException          if the argument is {@code null}.
    * @throws UnsupportedOperationException if the implementation of the argument is not supported.
    */
   @Override
   public void set(Matrix original)
   {
      if (original instanceof NativeMatrix)
         set((NativeMatrix) original);
      else if (original instanceof DMatrixRMaj)
         set((DMatrixRMaj) original);
      else if (original == null)
         throw new NullPointerException();
      else
         throw new UnsupportedOperationException("Unsupported matrix type: " + original.getClass().getSimpleName());
   }

   @Override
   public MatrixType getType()
   {
      return MatrixType.UNSPECIFIED;
   }

   /**
    * Unsafe get an element at row,col. If the index is out of bounds, Double.NaN is returned.
    */
   @Override
   public double unsafe_get(int row, int col)
   {
      return impl.get(row, col);
   }

   /**
    * Redirects to {@link #set(int, int, double)}.
    */
   @Override
   public void unsafe_set(int row, int col, double value)
   {
      set(row, col, value);
   }
}
