package us.ihmc.matrixlib;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.ops.CommonOps;
import org.ejml.ops.RandomMatrices;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.Conversions;

public class NativeCommonOpsTest
{
   private static final int maxSize = 80;
   private static final int warmumIterations = 2000;
   private static final int iterations = 5000;
   private static final double epsilon = 1.0e-8;

   @Test
   public void testMult()
   {
      Random random = new Random(40L);

      System.out.println("Testing matrix multiplications with random matrices...");

      long nativeTime = 0;
      long ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DenseMatrix64F A = RandomMatrices.createRandom(maxSize, maxSize, random);
         DenseMatrix64F B = RandomMatrices.createRandom(maxSize, maxSize, random);
         DenseMatrix64F AB = new DenseMatrix64F(maxSize, maxSize);
         CommonOps.mult(A, B, AB);
         NativeCommonOps.mult(A, B, AB);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         int bCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols + bCols) / 3.0;

         DenseMatrix64F A = RandomMatrices.createRandom(aRows, aCols, random);
         DenseMatrix64F B = RandomMatrices.createRandom(aCols, bCols, random);
         DenseMatrix64F actual = new DenseMatrix64F(aRows, bCols);
         DenseMatrix64F expected = new DenseMatrix64F(aRows, bCols);

         nativeTime -= System.nanoTime();
         NativeCommonOps.mult(A, B, actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps.mult(A, B, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
   }

   @Test
   public void testMultQuad()
   {
      Random random = new Random(40L);

      System.out.println("Testing computing quadratic form with random matrices...");

      long nativeTime = 0;
      long ejmlTime = 0;
      double matrixSizes = 0.0;

      for (int i = 0; i < warmumIterations; i++)
      {
         DenseMatrix64F A = RandomMatrices.createRandom(maxSize, maxSize, random);
         DenseMatrix64F B = RandomMatrices.createRandom(maxSize, maxSize, random);
         DenseMatrix64F tempBA = new DenseMatrix64F(maxSize, maxSize);
         DenseMatrix64F AtBA = new DenseMatrix64F(maxSize, maxSize);
         CommonOps.mult(B, A, tempBA);
         CommonOps.multTransA(A, tempBA, AtBA);
         NativeCommonOps.multQuad(A, B, AtBA);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         int aCols = random.nextInt(maxSize) + 1;
         matrixSizes += (aRows + aCols) / 2.0;

         DenseMatrix64F A = RandomMatrices.createRandom(aRows, aCols, random);
         DenseMatrix64F B = RandomMatrices.createRandom(aRows, aRows, random);
         DenseMatrix64F actual = new DenseMatrix64F(aCols, aCols);
         DenseMatrix64F expected = new DenseMatrix64F(aCols, aCols);
         DenseMatrix64F tempBA = new DenseMatrix64F(aRows, aCols);

         nativeTime -= System.nanoTime();
         NativeCommonOps.multQuad(A, B, actual);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         CommonOps.mult(B, A, tempBA);
         CommonOps.multTransA(A, tempBA, expected);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(expected, actual, epsilon);
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
   }

   @Test
   public void testInvert()
   {
      Random random = new Random(40L);

      System.out.println("Testing inverting with random matrices...");

      long nativeTime = 0;
      long ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.lu(maxSize);

      for (int i = 0; i < warmumIterations; i++)
      {
         DenseMatrix64F A = RandomMatrices.createRandom(maxSize, maxSize, -100.0, 100.0, random);
         DenseMatrix64F B = new DenseMatrix64F(maxSize, maxSize);
         solver.setA(A);
         solver.invert(B);
         NativeCommonOps.invert(A, B);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DenseMatrix64F A = RandomMatrices.createRandom(aRows, aRows, -100.0, 100.0, random);
         DenseMatrix64F nativeResult = new DenseMatrix64F(aRows, aRows);
         DenseMatrix64F ejmlResult = new DenseMatrix64F(aRows, aRows);

         nativeTime -= System.nanoTime();
         NativeCommonOps.invert(A, nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         solver.setA(A);
         solver.invert(ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(ejmlResult, nativeResult, epsilon);
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
   }

   @Test
   public void testSolve()
   {
      Random random = new Random(40L);

      System.out.println("Testing solving linear equations with random matrices...");

      long nativeTime = 0;
      long ejmlTime = 0;
      double matrixSizes = 0;
      LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.lu(maxSize);

      for (int i = 0; i < warmumIterations; i++)
      {
         DenseMatrix64F A = RandomMatrices.createRandom(maxSize, maxSize, random);
         DenseMatrix64F x = RandomMatrices.createRandom(maxSize, 1, random);
         DenseMatrix64F b = new DenseMatrix64F(maxSize, 1);
         CommonOps.mult(A, x, b);
         solver.setA(A);
         solver.solve(b, x);
         NativeCommonOps.solve(A, b, x);
      }

      for (int i = 0; i < iterations; i++)
      {
         int aRows = random.nextInt(maxSize) + 1;
         matrixSizes += aRows;

         DenseMatrix64F A = RandomMatrices.createRandom(aRows, aRows, random);
         DenseMatrix64F x = RandomMatrices.createRandom(aRows, 1, random);
         DenseMatrix64F b = new DenseMatrix64F(aRows, 1);
         CommonOps.mult(A, x, b);

         DenseMatrix64F nativeResult = new DenseMatrix64F(aRows, 1);
         DenseMatrix64F ejmlResult = new DenseMatrix64F(aRows, 1);

         nativeTime -= System.nanoTime();
         NativeCommonOps.solve(A, b, nativeResult);
         nativeTime += System.nanoTime();

         ejmlTime -= System.nanoTime();
         solver.setA(A);
         solver.solve(b, ejmlResult);
         ejmlTime += System.nanoTime();

         MatrixTestTools.assertMatrixEquals(x, nativeResult, epsilon);
         MatrixTestTools.assertMatrixEquals(x, ejmlResult, epsilon);
      }

      System.out.println("Native took " + Conversions.nanosecondsToMilliseconds((double) (nativeTime / iterations)) + " ms on average");
      System.out.println("EJML took " + Conversions.nanosecondsToMilliseconds((double) (ejmlTime / iterations)) + " ms on average");
      System.out.println("Average matrix size was " + matrixSizes / iterations);
      System.out.println("Native takes " + 100.0 * nativeTime / ejmlTime + "% of EJML time.\n");
   }

   public static void main(String[] args)
   {
      int size = 500;
      Random random = new Random(40L);
      DenseMatrix64F A = RandomMatrices.createRandom(size, size, random);
      DenseMatrix64F B = RandomMatrices.createRandom(size, size, random);
      DenseMatrix64F AtBA = new DenseMatrix64F(size, size);

      System.out.println("Running...");

      while (true)
      {
         NativeCommonOps.multQuad(A, B, AtBA);
      }
   }
}
