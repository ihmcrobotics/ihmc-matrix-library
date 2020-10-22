package us.ihmc.matrixlib;

import org.ejml.data.DMatrixRMaj;

import us.ihmc.matrixlib.jni.NativeMatrixImpl;
import us.ihmc.tools.nativelibraries.NativeLibraryLoader;

public class NativeMatrix
{
   static
   {
      NativeLibraryLoader.loadLibrary("", "NativeCommonOps");
   }

   private final NativeMatrixImpl impl;


   public NativeMatrix(int rows, int cols)
   {
      this.impl = new NativeMatrixImpl(rows, cols);
   }

   public NativeMatrix(DMatrixRMaj matrix)
   {
      this(matrix.getNumRows(), matrix.getNumCols());
      set(matrix);
   }
   
   public NativeMatrix(NativeMatrix matrix)
   {
      this(matrix.getNumRows(), matrix.getNumCols());
      set(matrix);
   }
   
   public void reshape(int rows, int cols)
   {
      impl.resize(rows, cols);
   }


   public void scale(double alpha, DMatrixRMaj matrix)
   {
      set(matrix);
      scale(alpha);
   }
   
   public void scale(double alpha, NativeMatrix matrix)
   {
      if(!impl.scale(alpha, matrix.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   public void set(DMatrixRMaj matrix)
   {
      if(!impl.set(matrix.data, matrix.numRows, matrix.numCols))
      {
         throw new IllegalArgumentException("Cannot set matrix.");
      }
   }
   
   public void set(NativeMatrix matrix)
   {
      if(!impl.set(matrix.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   public void get(DMatrixRMaj matrix)
   {      
      matrix.reshape(getNumRows(), getNumCols());
      
      if(!impl.get(matrix.data, matrix.numRows, matrix.numCols))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      
   }
   

   /**
    * Computes the matrix addition</br>
    * this = a + b
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * Computes the matrix subtraction</br>
    * this = a + b
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void mult(double scale, NativeMatrix a, NativeMatrix b)
   {
      
      if (!impl.mult(scale, a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public void multAdd(NativeMatrix a, NativeMatrix b)
   {
      
      if (!impl.multAdd(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   public void multAddTransA(NativeMatrix a, NativeMatrix b)
   {
      
      if (!impl.multAddTransA(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   public void multAddTransB(NativeMatrix a, NativeMatrix b)
   {
      
      if (!impl.multAddTransB(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public void multAddBlock(NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      
      if (!impl.multAddBlock(a.impl, b.impl, rowStart, colStart))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public void addBlock(NativeMatrix a, int destStartRow, int destStartColumn, int srcStartRow, int srcStartColumn, int numberOfRows, int numberOfColumns, double scale)
   {
      if(!impl.addBlock(a.impl, destStartRow, destStartColumn, srcStartRow, srcStartColumn, numberOfRows, numberOfColumns, scale))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the matrix multiplication</br>
    * this = a * b'
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * this = a' * b
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * Computes the quadratic form</br>
    * this = a' * b * a
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
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
    * Inverts a matrix.</br>
    * This method requires that the matrix is square and invertible and uses a LU decomposition.
    * 
    * @param a   matrix to invert
    * @param inv where the result is stored (modified)
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
   
   public void transpose(NativeMatrix a)
   {
      if(!impl.transpose(a.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * this == b</br>
    * This method requires that the matrix a is square and invertible and uses a LU decomposition.
    * 
    * @param a matrix in equation
    * @param b matrix in equation
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public void solve(NativeMatrix a, NativeMatrix b)
   {

      if (!impl.solve(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public boolean solveCheck(NativeMatrix a, NativeMatrix b)
   {
           
      if(impl.solveCheck(a.impl, b.impl))
      {
         return true;
      }
      else
      {
         return false;
      }
      

   }

   public void insert(NativeMatrix src, int srcY0, int srcY1, int srcX0, int srcX1, int dstY0, int dstX0)
   {
      if (!impl.insert(src.impl, srcY0, srcY1, srcX0, srcX1, dstY0, dstX0))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public void insert(NativeMatrix src, int destY0, int destX0)
   {
      insert(src, 0, src.getNumRows(), 0, src.getNumCols(), destY0, destX0);
   }
   
   public double get(int row, int col)
   {
      return impl.get(row, col);
   }
   
   public void set(int row, int col, double value)
   {
      if(!impl.set(row, col, value))
      {
         throw new IllegalArgumentException("Index out of bounds. Requested (" + row + ", " + col + "). Dimension (" + getNumRows() + ", " + getNumCols() + ").");
      }
   }
   
   public void removeRow(int row)
   {
      if(!impl.removeRow(row))
      {
         throw new IllegalArgumentException("Row out of bounds.");
      }
   }
   

   public void removeColumn(int col)
   {
      if(!impl.removeColumn(col))
      {
         throw new IllegalArgumentException("Col out of bounds.");
      }
   }
   
   public void zero()
   {
      impl.zero();
   }
   
   public boolean containsNaN()
   {
      return impl.containsNaN();
   }

   public int getNumRows()
   {
      return impl.rows();
   }

   public int getNumCols()
   {
      return impl.cols();
   }
   
   public double min()
   {
      return impl.min();
   }
   
   public double max()
   {
      return impl.max();
   }
   
   public double sum()
   {
      return impl.sum();
   }
   
   public double prod()
   {
      return impl.prod();
   }
   
   public void scale(double scale)
   {
      impl.scale(scale);
   }

   public void print()
   {
      impl.print();
   }

   public boolean isApprox(NativeMatrix solution, double precision)
   {
      return impl.isAprrox(solution.impl, precision);
   }

   public String toString()
   {
      return "NativeMatrix: " + getNumRows() + " x " + getNumCols();
   }


}
