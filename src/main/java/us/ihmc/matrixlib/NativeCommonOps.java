package us.ihmc.matrixlib;

import org.ejml.data.DMatrix1Row;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.robotics.linearAlgebra.commonOps.NativeCommonOpsWrapper;
import us.ihmc.tools.nativelibraries.NativeLibraryLoader;

public class NativeCommonOps
{
   private static final NativeCommonOpsWrapper nativeCommonOpsWrapper = loadNativeOps();

   public static NativeCommonOpsWrapper loadNativeOps()
   {
      NativeLibraryLoader.loadLibrary("", "NativeCommonOps");
      return new NativeCommonOpsWrapper();
   }

   /**
    * Computes the matrix multiplication</br>
    * c = a * b
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
    * @param c where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void mult(DMatrix1Row a, DMatrix1Row b, DMatrix1Row c)
   {
      if (a.getNumCols() != b.getNumRows())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      c.reshape(a.getNumRows(), b.getNumCols());
      nativeCommonOpsWrapper.mult(c.data, a.data, b.data, a.getNumRows(), a.getNumCols(), b.getNumCols());
   }

   /**
    * Computes the quadratic form</br>
    * c = a' * b * a
    * 
    * @param a matrix in multiplication
    * @param b matrix in multiplication
    * @param c where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void multQuad(DMatrix1Row a, DMatrix1Row b, DMatrix1Row c)
   {
      if (a.getNumRows() != b.getNumCols() || b.getNumCols() != b.getNumRows())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      c.reshape(a.getNumCols(), a.getNumCols());
      nativeCommonOpsWrapper.multQuad(c.data, a.data, b.data, a.getNumRows(), a.getNumCols());
   }

   /**
    * Inverts a matrix.</br>
    * This method requires that the matrix is square and invertible and uses a LU decomposition.
    * 
    * @param a   matrix to invert
    * @param inv where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void invert(DMatrix1Row a, DMatrix1Row inv)
   {
      if (a == inv)
      {
         throw new IllegalArgumentException("Can not invert in place. The result matrix needs to be different from the matrix to invert.");
      }
      if (a.getNumRows() != a.getNumCols())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      inv.reshape(a.getNumRows(), a.getNumCols());
      nativeCommonOpsWrapper.invert(inv.data, a.data, a.getNumRows());
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * x == b</br>
    * This method requires that the matrix a is square and invertible and uses a LU decomposition.
    * 
    * @param a matrix in equation
    * @param b matrix in equation
    * @param x where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void solve(DMatrix1Row a, DMatrix1Row b, DMatrix1Row x)
   {
      if (a.getNumRows() != b.getNumRows() || b.getNumCols() != 1 || a.getNumCols() != a.getNumRows())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      x.reshape(a.getNumCols(), 1);
      nativeCommonOpsWrapper.solve(x.data, a.data, b.data, a.getNumRows());
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * x == b</br>
    * This method requires that {@code a} is square. It will check the invertability of {@code a} and
    * will return false if it is not invertible.
    * 
    * @param a matrix in equation
    * @param b matrix in equation
    * @param x where the result is stored (modified)
    * @return whether a solution was found
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static boolean solveCheck(DMatrix1Row a, DMatrix1Row b, DMatrix1Row x)
   {
      if (a.getNumRows() != b.getNumRows() || b.getNumCols() != 1 || a.getNumCols() != a.getNumRows())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      x.reshape(a.getNumCols(), 1);
      if (nativeCommonOpsWrapper.solveCheck(x.data, a.data, b.data, a.getNumRows()))
      {
         return true;
      }
      CommonOps_DDRM.fill(x, Double.NaN);
      return false;
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * x == b</br>
    * This method does not require that the matrix a is square and invertible and can be used to solve
    * a least square problem. It uses a householder QR decomposition.
    * 
    * @param a matrix in equation
    * @param b matrix in equation
    * @param x where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void solveRobust(DMatrix1Row a, DMatrix1Row b, DMatrix1Row x)
   {
      if (a.getNumRows() != b.getNumRows() || b.getNumCols() != 1)
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      x.reshape(a.getNumCols(), 1);
      nativeCommonOpsWrapper.solveRobust(x.data, a.data, b.data, a.getNumRows(), a.getNumCols());
   }

   /**
    * Computes the solution to the linear equation</br>
    * a * x == b</br>
    * This method uses a damped least square approach and a Cholesky decomposition of</br>
    * a * a' + diag(alpha * alpha)
    * 
    * @param a     matrix in equation
    * @param b     matrix in equation
    * @param alpha damping value
    * @param x     where the result is stored (modified)
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void solveDamped(DMatrix1Row a, DMatrix1Row b, double alpha, DMatrix1Row x)
   {
      if (a.getNumRows() != b.getNumRows() || b.getNumCols() != 1)
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      x.reshape(a.getNumCols(), 1);
      nativeCommonOpsWrapper.solveDamped(x.data, a.data, b.data, a.getNumRows(), a.getNumCols(), alpha);
   }

   /**
    * Projects the matrix {@code a} onto the null-space of {@code b} and stores the result in {@code c}
    * such that</br>
    * b * c == 0</br>
    * This method uses a damped least square approach causing the null-space to grow gradually.
    * 
    * @param a     matrix to project
    * @param b     matrix to compute the null-space of
    * @param c     where the result is stored (modified)
    * @param alpha damping value
    * @throws IllegalArgumentException if the matrix dimensions are incompatible.
    */
   public static void projectOnNullspace(DMatrix1Row a, DMatrix1Row b, DMatrix1Row c, double alpha)
   {
      if (a == c)
      {
         throw new IllegalArgumentException("Can not project in place. The result matrix needs to be different from the matrix to project.");
      }
      if (a.getNumCols() != b.getNumCols())
      {
         throw new IllegalArgumentException("Incompatible Matrix Dimensions.");
      }
      c.reshape(a.getNumRows(), a.getNumCols());
      nativeCommonOpsWrapper.projectOnNullspace(c.data, a.data, b.data, a.getNumRows(), a.getNumCols(), b.getNumRows(), alpha);
   }
}
