package us.ihmc.matrixlib;

import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.Random;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.RandomMatrices_DDRM;

import us.ihmc.matrixlib.jni.NativeMatrixImpl;
import us.ihmc.tools.nativelibraries.NativeLibraryLoader;

public class NativeMatrix
{
   static
   {
      NativeLibraryLoader.loadLibrary("", "NativeCommonOps");
   }

   private final NativeMatrixImpl impl;

   private int rows;
   private int cols;
   private DoubleBuffer data;

   public NativeMatrix(int rows, int cols)
   {
      this.impl = new NativeMatrixImpl(rows, cols);
      update();
   }

   private void update()
   {
      this.rows = impl.rows();
      this.cols = impl.cols();
      this.data = impl.data().order(ByteOrder.nativeOrder()).asDoubleBuffer();
   }

   public void scale(double alpha, DMatrix matrix)
   {
      if (matrix.getNumCols() != cols || matrix.getNumRows() != rows)
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }

      data.clear();
      for (int c = 0; c < matrix.getNumCols(); c++)
      {
         for (int r = 0; r < matrix.getNumRows(); r++)
         {
            data.put(alpha * matrix.unsafe_get(r, c));
         }
      }
   }
   
   public void scale(double alpha, NativeMatrix matrix)
   {
      if(!impl.scale(alpha, matrix.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }

   public void set(DMatrix matrix)
   {
      if (matrix.getNumCols() != cols || matrix.getNumRows() != rows)
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }

      data.clear();
      for (int c = 0; c < matrix.getNumCols(); c++)
      {
         for (int r = 0; r < matrix.getNumRows(); r++)
         {
            data.put(matrix.unsafe_get(r, c));
         }
      }
   }

   public void get(DMatrix matrix)
   {
      if (matrix.getNumCols() != cols || matrix.getNumRows() != rows)
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }

      data.clear();
      for (int c = 0; c < matrix.getNumCols(); c++)
      {
         for (int r = 0; r < matrix.getNumRows(); r++)
         {
            matrix.unsafe_set(r, c, data.get());
         }
      }
   }

   public void reshape(int rows, int cols)
   {
      resize(rows, cols);
   }

   public void resize(int rows, int cols)
   {
      if (rows == this.rows && cols == this.cols)
      {
         return;
      }

      System.err.println("Resizing matrix to " + rows + " " + cols);

      impl.resize(rows, cols);
      update();
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
      resize(a.getNumRows(), b.getNumCols());

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
      resize(a.getNumRows(), b.getNumCols());
      
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
   
   public void multAddBlock(NativeMatrix a, NativeMatrix b, int rowStart, int colStart)
   {
      
      if (!impl.multAddBlock(a.impl, b.impl, rowStart, colStart))
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
      resize(a.getNumRows(), b.getNumRows());

      if (!impl.multTransB(a.impl, b.impl))
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
      resize(a.getNumCols(), a.getNumCols());

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

      resize(a.getNumRows(), a.getNumCols());
      if (!impl.invert(a.impl))
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
      resize(a.getNumCols(), 1);

      if (!impl.solve(a.impl, b.impl))
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
   }
   
   public boolean solveCheck(NativeMatrix a, NativeMatrix b)
   {
      resize(a.getNumCols(), 1);
      
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
      return data.get(col * cols + row);
   }
   
   public void set(int row, int col, double value)
   {
      data.put(col * cols + row, value);
   }
   
   public void zero()
   {
      impl.zero();
   }

   public int getNumRows()
   {
      return rows;
   }

   public int getNumCols()
   {
      return cols;
   }

   public void print()
   {
      impl.print();
   }

   public static void main(String[] args)
   {
      NativeMatrix m = new NativeMatrix(5, 10);

      System.out.println("N---");
      m.print();
      System.out.println("N---");
      DMatrixRMaj A = RandomMatrices_DDRM.rectangle(5, 10, new Random());
      m.set(A);
      System.out.println("J---");
      A.print();
      System.out.println("J---");
      System.out.println("N---");
      m.print();
      System.out.println("N---");

      System.out.println(m.data);
   }

}
